#!/bin/bash

#SBATCH --job-name=16bands_cloudphase                          # Job name
#SBATCH --output=cloudphase16_test_%u_%j.out             # Output file name
#SBATCH --error=cloudphase16_test_%u_%j.err              # Error file name
#SBATCH --account=cybertrn                                  # Account name
#SBATCH --cluster=chip-gpu                                              # specify gpu or cpu
#SBATCH --mem=220G                                          # Job memory request 
#SBATCH --constraint="L40S|H100"
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-02:30:00      

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Loading conda environment..."
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate /umbc/rs/nasa-access/users/xingyan/software/miniconda3/envs/satvision

srun python run_phase_pred.py --config-path configs/16bands.yaml
