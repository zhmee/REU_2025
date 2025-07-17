#!/bin/bash

#SBATCH --job-name=mlp32_withtransform            # Job name
#SBATCH --output=mlp32_transform_%j.out             # Output file name
#SBATCH --error=mlp32_transform_%j.err              # Error file name
#SBATCH --account=cybertrn                                  # Account name
#SBATCH --cluster=chip-gpu                                              # specify gpu or cpu
#SBATCH --mem=220G                                          # Job memory request 
#SBATCH --constraint="L40S|H100"
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-02:00:00                                   # Time limit days-hrs:min:sec

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Loading conda environment..."
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate /umbc/rs/nasa-access/users/xingyan/software/miniconda3/envs/satvision

echo "Disk usage:"
df -h

echo "Disk usage in home:"
du -sh ~/* | sort -hr | head

srun python run_phase_pred.py --config-path configs/16bands.yaml
