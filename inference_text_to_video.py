import torch
import numpy as np
print(torch.cuda.is_available())  # 應該返回 True
print(torch.version.cuda)          # 應該顯示 12.4
print(torch.cuda.get_device_name(torch.cuda.current_device())) # GPU名稱
from model import Model
model = Model(device = "cuda", dtype = torch.float16)
from hf_utils import get_model_list
model_list = get_model_list()
# for idx, name in enumerate(model_list):
#   print(idx, name)
# prompt = "A horse galloping on a street"
print(f'model name={model_list[3]}')
# prompt = "A chinese landscape painting of a boat drifting in a river stream at the foot of mist-surronded green moutains. The with fluffy clouds floats by with elegant birds taking flight."
prompt = "A chinese landscape painting of a boat drifting in a river stream at the foot of mist-surronded green moutains. The foreground object is the boat and the background objects are the river and the mountains. The foreground object should appear in every frame."
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 1, "motion_field_strength_y" : 1, "video_length": 90,"model_name":  model_list[3],"smooth_bg": True}
# params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 1, "motion_field_strength_y" : 1, "video_length": 8, "chunk_size": 4, "model_name":  model_list[3]}
# out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
out_path, fps = f"/home/yccra/T2Vv2/text2video_landscape.mp4", 30
_,scores=model.process_text2video(prompt, fps = fps, path = out_path, **params)
np.save("/home/yccra/T2Vv2/scores.npy", scores)
