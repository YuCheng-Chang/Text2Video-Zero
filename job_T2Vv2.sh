#!/bin/bash
#SBATCH -J My-Test-Job # Job Name
#SBATCH -o my_test_job.out # SLURM standard output to file
#SBATCH -e my_test_job.err # SLURM standard error to file
#SBATCH --nodes=1 # Utilize 1 nodes for this job
#SBATCH --ntasks-per-node=1 # Each node runs 4 tasks, so two nodes will totally run 8 tasks
#SBATCH --cpus-per-task=1 # Each task reserves 8 vcores on a node, default value is 1
#SBATCH --gres=gpu:1 # Reserve 4 GPUs on each node for this job
#SBATCH --mem=20G # Reserve memory size on each node
#SBATCH --time=01:00:00 # Set execution time limit to 3 mins, kill the job if it reaches
#SBATCH -p defq # Partition/Queue name

#==========================
# Load modules
#==========================

module purge
module load slurm/slurm/23.02.4
module load nvidia-hpc/2024_241
module load nvhpc/24.1
module list

#==========================
# Execute My Program
#==========================

# srun awk 'BEGIN { print "Hello from " ENVIRON["SLURMD_NODENAME"] ", my SLURM_PROCID is " ENVIRON["SLURM_PROCID"] "." }'
srun python /home/yccra/T2Vv2/inference_text_to_video.py