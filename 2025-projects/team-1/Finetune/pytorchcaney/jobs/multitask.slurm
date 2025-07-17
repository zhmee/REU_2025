#!/bin/bash

#SBATCH --job-name=Multitask            # Job name
#SBATCH --output=slurm_logs/Multitask/FFT/%j.out             # Output file name
#SBATCH --error=slurm_logs/Multitask/FFT/%j.err              # Error file name
#SBATCH --account=pi_jianwu  			# Account name
#SBATCH --reservation=jianwu
#SBATCH --qos=chip-gpu
#SBATCH --cluster=chip-gpu                                              # specify gpu or cpu
#SBATCH --mem=120G                                          # Job memory request 
#SBATCH --constraint="L40S|H100"
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-02:30:00                                   # Time limit days-hrs:min:sec

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Loading conda environment..."
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate /umbc/rs/nasa-access/users/xingyan/software/miniconda3/envs/satvision
source env_local.sh
echo "$DS_2D_CLOUD_ROOT"
srun python run_cloud_pred.py --config-path configs/multitask.yaml
