from diffusers import StableDiffusionPipeline
import torch
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import numpy as np
from diffusers.utils import deprecate, logging, BaseOutput
from einops import rearrange, repeat
from torch.nn.functional import grid_sample
import torchvision.transforms as T
from torchvision.utils import save_image
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import PIL
from PIL import Image
from kornia.morphology import dilation
import sys
import os
from ultralytics import YOLO


# 獲取當前文件的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
# 構建 U2-net-master 的絕對路徑
u2net_path = os.path.join(current_dir, "U2_net_master")
# 將路徑添加到 sys.path
sys.path.append(u2net_path)

# 打印 sys.path 以進行調試
print("sys.path:", sys.path)


# 然後導入
from U2_net_master.model.u2net import U2NETP, U2NET
from U2_net_master.model.preprocessing import *

@dataclass
class TextToVideoPipelineOutput(BaseOutput):
    # videos: Union[torch.Tensor, np.ndarray]
    # code: Union[torch.Tensor, np.ndarray]
    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


def coords_grid(batch, ht, wd, device):
    # Adapted from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    coords = torch.meshgrid(torch.arange(
        ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class TextToVideoPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         safety_checker, feature_extractor, requires_safety_checker)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'current device : {self._device}')
        self.sod_model = YOLO("yolo11n-seg.pt")  # load an official model

        self.sod_model = self.sod_model.to(self._device)
        # self.sod_model.eval()
        # self.sod_transform = T.Compose([RescaleT(320),ToTensorLab(flag=0)])
        self.save_mask_idx=0
        self.save_z0f_idx=0

    def DDPM_forward(self, x0, t0, tMax, generator, device, shape, text_embeddings):
        rand_device = "cpu" if device.type == "mps" else device

        if x0 is None:
            return torch.randn(shape, generator=generator, device=rand_device, dtype=text_embeddings.dtype).to(device)
        else:
            eps = torch.randn(x0.shape, dtype=text_embeddings.dtype, generator=generator,
                              device=rand_device)
            alpha_vec = torch.prod(self.scheduler.alphas[t0:tMax])

            xt = torch.sqrt(alpha_vec) * x0 + \
                torch.sqrt(1-alpha_vec) * eps
            return xt

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height //
                 self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(
                        shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(
                    shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def warp_latents_independently(self, latents, reference_flow):
        _, _, H, W = reference_flow.size()
        b, _, f, h, w = latents.size()
        assert b == 1
        coords0 = coords_grid(f, H, W, device=latents.device).to(latents.dtype)

        coords_t0 = coords0 + reference_flow
        coords_t0[:, 0] /= W
        coords_t0[:, 1] /= H

        coords_t0 = coords_t0 * 2.0 - 1.0

        coords_t0 = T.Resize((h, w))(coords_t0)

        coords_t0 = rearrange(coords_t0, 'f c h w -> f h w c')

        latents_0 = rearrange(latents[0], 'c f h w -> f  c  h w')
        warped = grid_sample(latents_0, coords_t0,
                             mode='nearest', padding_mode='reflection')

        warped = rearrange(warped, '(b f) c h w -> b c f h w', f=f)
        return warped

    def DDIM_backward(self, num_inference_steps, timesteps, skip_t, t0, t1, do_classifier_free_guidance, null_embs, text_embeddings, latents_local,
                      latents_dtype, guidance_scale, guidance_stop_step, callback, callback_steps, extra_step_kwargs, num_warmup_steps):
        entered = False

        f = latents_local.shape[2]

        latents_local = rearrange(latents_local, "b c f w h -> (b f) c w h")

        latents = latents_local.detach().clone()
        x_t0_1 = None
        x_t1_1 = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if t > skip_t:
                    continue
                else:
                    if not entered:
                        print(
                            f"Continue DDIM with i = {i}, t = {t}, latent = {latents.shape}, device = {latents.device}, type = {latents.dtype}")
                        entered = True

                latents = latents.detach()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    if null_embs is not None:
                        text_embeddings[0] = null_embs[i][0]
                    te = torch.cat([repeat(text_embeddings[0, :, :], "c k -> f c k", f=f),
                                   repeat(text_embeddings[1, :, :], "c k -> f c k", f=f)])
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=te).sample.to(dtype=latents_dtype)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(
                        2)
                    noise_pred = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                if i >= guidance_stop_step * len(timesteps):
                    alpha = 0
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # latents = latents - alpha * grads / (torch.norm(grads) + 1e-10)
                # call the callback, if provided

                if i < len(timesteps)-1 and timesteps[i+1] == t0:
                    x_t0_1 = latents.detach().clone()
                    print(f"latent t0 found at i = {i}, t = {t}")
                elif i < len(timesteps)-1 and timesteps[i+1] == t1:
                    x_t1_1 = latents.detach().clone()
                    print(f"latent t1 found at i={i}, t = {t}")

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        latents = rearrange(latents, "(b f) c w h -> b c f  w h", f=f)

        res = {"x0": latents.detach().clone()}
        if x_t0_1 is not None:
            x_t0_1 = rearrange(x_t0_1, "(b f) c w h -> b c f  w h", f=f)
            res["x_t0_1"] = x_t0_1.detach().clone()
        if x_t1_1 is not None:
            x_t1_1 = rearrange(x_t1_1, "(b f) c w h -> b c f  w h", f=f)
            res["x_t1_1"] = x_t1_1.detach().clone()
        return res

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.detach().cpu()
        return video

    def create_motion_field(self, motion_field_strength_x, motion_field_strength_y, frame_ids, video_length, latents):

        reference_flow = torch.zeros(
            (video_length-1, 2, 512, 512), device=latents.device, dtype=latents.dtype)
        for fr_idx, frame_id in enumerate(frame_ids):
            reference_flow[fr_idx, 0, :,
                           :] = motion_field_strength_x*(frame_id)
            reference_flow[fr_idx, 1, :,
                           :] = motion_field_strength_y*(frame_id)
        return reference_flow

    def create_motion_field_and_warp_latents(self, motion_field_strength_x, motion_field_strength_y, frame_ids, video_length, latents):

        motion_field = self.create_motion_field(motion_field_strength_x=motion_field_strength_x,
                                                motion_field_strength_y=motion_field_strength_y, latents=latents, video_length=video_length, frame_ids=frame_ids)
        for idx, latent in enumerate(latents):
            latents[idx] = self.warp_latents_independently(
                latent[None], motion_field)
        return motion_field, latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_stop_step: float = 0.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        xT: Optional[torch.FloatTensor] = None,
        null_embs: Optional[torch.FloatTensor] = None,
        motion_field_strength_x: float = 12,
        motion_field_strength_y: float = 12,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[
            int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        use_motion_field: bool = True,
        smooth_bg: bool = False,
        smooth_bg_strength: float = 0.4,
        t0: int = 44,
        t1: int = 47,
        **kwargs,
    ):
        frame_ids = kwargs.pop("frame_ids", list(range(video_length)))
        assert t0 < t1
        assert num_videos_per_prompt == 1
        assert isinstance(prompt, list) and len(prompt) > 0
        assert isinstance(negative_prompt, list) or negative_prompt is None

        prompt_types = [prompt, negative_prompt]

        for idx, prompt_type in enumerate(prompt_types):
            prompt_template = None
            for prompt in prompt_type:
                if prompt_template is None:
                    prompt_template = prompt
                else:
                    assert prompt == prompt_template
            if prompt_types[idx] is not None:
                prompt_types[idx] = prompt_types[idx][0]
        prompt = prompt_types[0]
        negative_prompt = prompt_types[1]

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # print(f" Latent shape = {latents.shape}")

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels

        xT = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            1,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            xT,
        )#size: bclhw
        dtype = xT.dtype

        # when motion field is not used, augment with random latent codes
        if use_motion_field:
            xT = xT[:, :, :1]
        else:
            if xT.shape[2] < video_length:
                xT_missing = self.prepare_latents(
                    batch_size * num_videos_per_prompt,
                    num_channels_latents,
                    video_length-xT.shape[2],
                    height,
                    width,
                    text_embeddings.dtype,
                    device,
                    generator,
                    None,
                )
                xT = torch.cat([xT, xT_missing], dim=2)

        xInit = xT.clone()
        # print(f'xT shape: {xT.shape}')#xT shape: torch.Size([1, 4, 1, 64, 64])
        timesteps_ddpm = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741, 721,
                          701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461, 441,
                          421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181, 161,
                          141, 121, 101,  81,  61,  41,  21,   1]
        timesteps_ddpm.reverse()

        t0 = timesteps_ddpm[t0]
        t1 = timesteps_ddpm[t1]

        print(f"t0 = {t0} t1 = {t1}")
        x_t1_1 = None

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # Denoising loop
        num_warmup_steps = len(timesteps) - \
            num_inference_steps * self.scheduler.order

        shape = (batch_size, num_channels_latents, 1, height //
                 self.vae_scale_factor, width // self.vae_scale_factor)

        ddim_res = self.DDIM_backward(num_inference_steps=num_inference_steps, timesteps=timesteps, skip_t=1000, t0=t0, t1=t1, do_classifier_free_guidance=do_classifier_free_guidance,
                                      null_embs=null_embs, text_embeddings=text_embeddings, latents_local=xT, latents_dtype=dtype, guidance_scale=guidance_scale, guidance_stop_step=guidance_stop_step,
                                      callback=callback, callback_steps=callback_steps, extra_step_kwargs=extra_step_kwargs, num_warmup_steps=num_warmup_steps)

        x0 = ddim_res["x0"].detach()

        if "x_t0_1" in ddim_res:
            x_t0_1 = ddim_res["x_t0_1"].detach()
        if "x_t1_1" in ddim_res:
            x_t1_1 = ddim_res["x_t1_1"].detach()
        del ddim_res
        
        del xT
        if use_motion_field:
            del x0

            x_t0_k = x_t0_1[:, :, :1, :, :].repeat(1, 1, video_length-1, 1, 1)

            reference_flow, x_t0_k = self.create_motion_field_and_warp_latents(
                motion_field_strength_x=motion_field_strength_x, motion_field_strength_y=motion_field_strength_y, latents=x_t0_k, video_length=video_length, frame_ids=frame_ids[1:])

            # assuming t0=t1=1000, if t0 = 1000
            if t1 > t0:
                x_t1_k = self.DDPM_forward(
                    x0=x_t0_k, t0=t0, tMax=t1, device=device, shape=shape, text_embeddings=text_embeddings, generator=generator)
            else:
                x_t1_k = x_t0_k

            if x_t1_1 is None:
                raise Exception

            x_t1 = torch.cat([x_t1_1, x_t1_k], dim=2).clone().detach()

            ddim_res = self.DDIM_backward(num_inference_steps=num_inference_steps, timesteps=timesteps, skip_t=t1, t0=-1, t1=-1, do_classifier_free_guidance=do_classifier_free_guidance,
                                          null_embs=null_embs, text_embeddings=text_embeddings, latents_local=x_t1, latents_dtype=dtype, guidance_scale=guidance_scale,
                                          guidance_stop_step=guidance_stop_step, callback=callback, callback_steps=callback_steps, extra_step_kwargs=extra_step_kwargs, num_warmup_steps=num_warmup_steps)

            x0 = ddim_res["x0"].detach()
            del ddim_res
            # del x_t1
            # del x_t1_1
            # del x_t1_k
            print(f'x_t1:{x_t1.shape} x_t1_1:{x_t1_1.shape} x_t1_k:{x_t1_k.shape}')
        else:
            x_t1 = x_t1_1.clone()
            x_t1_1 = x_t1_1[:, :, :1, :, :].clone()
            x_t1_k = x_t1_1[:, :, 1:, :, :].clone()
            x_t0_k = x_t0_1[:, :, 1:, :, :].clone()
            x_t0_1 = x_t0_1[:, :, :1, :, :].clone()

        # smooth background
        
        if smooth_bg:
            h, w = x0.shape[3], x0.shape[4]
            # print(f'(h,w) = {h},{w}')#(h,w) = (64,64)
            M_FG = torch.zeros((batch_size, video_length, h, w),
                               device=x0.device).to(x0.dtype)
            for batch_idx, x0_b in enumerate(x0):
                z0_b = self.decode_latents(x0_b[None]).detach()
                z0_b = rearrange(z0_b[0], "c f h w -> f h w c")
                # print(f'z0_b shape={z0_b.shape}')#(8, 512, 512, 3)
                for frame_idx, z0_f in enumerate(z0_b):
                    # 確保 z0_f 是在正確設備（CPU 或 GPU）上的 PyTorch 張量
                    z0_f = torch.tensor(z0_f, dtype=torch.float32, device=self.device)
                    
                    # 將值範圍調整到 0-1 之間
                    z0_f = (z0_f - z0_f.min()) / (z0_f.max() - z0_f.min())
                    
                    # 準備 YOLO 模型的輸入
                    # 1. 確保張量形狀為 (1, 3, 640, 640)
                    # 2. 調整通道順序從 HWC 到 CHW
                    # 3. 添加 batch 維度
                    z0_f_yolo = z0_f.permute(2, 0, 1).unsqueeze(0)
                    # 使用YOLO進行預測
                    with torch.no_grad():
                        results = self.sod_model(z0_f_yolo)  # 直接傳入numpy數組
                        
                        # 獲取預測結果
                        if len(results) > 0:
                            result = results[0]  # 獲取第一個（也是唯一的）結果
                            masks = result.masks  # 獲取所有實例的 mask
                            
                            if masks is not None and len(masks) > 0:
                                # 合併所有實例的 mask
                                combined_mask = masks.data.sum(dim=0) > 0
                                mask = combined_mask.float()  # 轉換為浮點數
                                
                                print(f"Mask shape after sod_model: {mask.shape}")
                                
                                # 調整 mask 大小
                                mask = T.Resize(size=(h, w), interpolation=T.InterpolationMode.NEAREST)(mask.unsqueeze(0)).squeeze(0)
                                
                                print(f"Mask shape after resize: {mask.shape}, mean: {torch.mean(mask)}")
                            else:
                                mask = torch.zeros((h, w), dtype=torch.float32)
                        else:
                            mask = torch.zeros((h, w), dtype=torch.float32)
                        
                    # 後處理預測結果
                    mask = (mask > 0.5).float()  # 二值化，閾值可以根據需要調整
                    # print(mask.shape)#(1,1,512,512)
                    # save_image(z0_f.cpu(),f'/home/yccra/T2Vv2/SOD_results/z0f_{self.save_z0f_idx}.png')
                    # save_image(mask, f'/home/yccra/T2Vv2/SOD_results/mask_{self.save_mask_idx}.png')
                    # 保存原始圖像
                    save_image(z0_f.permute(2, 0, 1), f'/home/yccra/T2Vv2/SOD_results/z0f_{self.save_z0f_idx}.png')
                    
                    # 保存 mask
                    save_image(mask.unsqueeze(0), f'/home/yccra/T2Vv2/SOD_results/mask_{self.save_mask_idx}.png')
                    self.save_mask_idx+=1
                    self.save_z0f_idx+=1
                    # 調整大小和應用膨脹
                    # mask = T.Resize(size=(h, w), interpolation=T.InterpolationMode.NEAREST)(mask[None])
                    # mask = T.Resize(size=(h, w), interpolation=T.InterpolationMode.NEAREST)(mask)
                    # kernel = torch.ones(5, 5, device=x0.device, dtype=x0.dtype)
                   # 調整大小和應用膨脹
                    mask = mask.unsqueeze(0).unsqueeze(0)  # 添加批次和通道維度
                    print(f"Mask shape before dilation: {mask.shape}")
                    kernel = torch.ones(5, 5, device=x0.device, dtype=x0.dtype)
                    mask = dilation(mask.to(x0.device), kernel)[0, 0]  # 移除批次和通道維度
                    # apply SOD detection
                    # m_f = torch.tensor(self.sod_model.process_data(
                    #     z0_f), device=x0.device).to(x0.dtype)
                    # mask = T.Resize(
                    #     size=(h, w), interpolation=T.InterpolationMode.NEAREST)(m_f[None])
                    # kernel = torch.ones(5, 5, device=x0.device, dtype=x0.dtype)
                    # mask = dilation(mask[None].to(x0.device), kernel)[0]
                    # 確保 mask 的形狀是 (h, w)
                    print(f"Mask shape after dilation: {mask.shape}")
                    M_FG[batch_idx, frame_idx, :, :] = mask

            x_t1_1_fg_masked = x_t1_1 * \
                (1 - repeat(M_FG[:, 0, :, :],
                            "b w h -> b c 1 w h", c=x_t1_1.shape[1]))

            x_t1_1_fg_masked_moved = []
            for batch_idx, x_t1_1_fg_masked_b in enumerate(x_t1_1_fg_masked):
                x_t1_fg_masked_b = x_t1_1_fg_masked_b.clone()

                x_t1_fg_masked_b = x_t1_fg_masked_b.repeat(
                    1, video_length-1, 1, 1)
                if use_motion_field:
                    x_t1_fg_masked_b = x_t1_fg_masked_b[None]
                    x_t1_fg_masked_b = self.warp_latents_independently(
                        x_t1_fg_masked_b, reference_flow)
                else:
                    x_t1_fg_masked_b = x_t1_fg_masked_b[None]

                x_t1_fg_masked_b = torch.cat(
                    [x_t1_1_fg_masked_b[None], x_t1_fg_masked_b], dim=2)
                x_t1_1_fg_masked_moved.append(x_t1_fg_masked_b)

            x_t1_1_fg_masked_moved = torch.cat(x_t1_1_fg_masked_moved, dim=0)

            M_FG_1 = M_FG[:, :1, :, :]

            M_FG_warped = []
            for batch_idx, m_fg_1_b in enumerate(M_FG_1):
                m_fg_1_b = m_fg_1_b[None, None]
                m_fg_b = m_fg_1_b.repeat(1, 1, video_length-1, 1, 1)
                if use_motion_field:
                    m_fg_b = self.warp_latents_independently(
                        m_fg_b.clone(), reference_flow)
                M_FG_warped.append(
                    torch.cat([m_fg_1_b[:1, 0], m_fg_b[:1, 0]], dim=1))

            M_FG_warped = torch.cat(M_FG_warped, dim=0)

            channels = x0.shape[1]

            M_BG = (1-M_FG) * (1 - M_FG_warped)
            M_BG = repeat(M_BG, "b f h w -> b c f h w", c=channels)
            a_convex = smooth_bg_strength

            latents = (1-M_BG) * x_t1 + M_BG * (a_convex *
                                                x_t1 + (1-a_convex) * x_t1_1_fg_masked_moved)

            ddim_res = self.DDIM_backward(num_inference_steps=num_inference_steps, timesteps=timesteps, skip_t=t1, t0=-1, t1=-1, do_classifier_free_guidance=do_classifier_free_guidance,
                                          null_embs=null_embs, text_embeddings=text_embeddings, latents_local=latents, latents_dtype=dtype, guidance_scale=guidance_scale,
                                          guidance_stop_step=guidance_stop_step, callback=callback, callback_steps=callback_steps, extra_step_kwargs=extra_step_kwargs, num_warmup_steps=num_warmup_steps)
            x0 = ddim_res["x0"].detach()
            del ddim_res
            del latents

        latents = x0

        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
        torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        else:
            image = self.decode_latents(latents)

            # Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, text_embeddings.dtype)
            image = rearrange(image, "b c f h w -> (b f) h w c")

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return TextToVideoPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
