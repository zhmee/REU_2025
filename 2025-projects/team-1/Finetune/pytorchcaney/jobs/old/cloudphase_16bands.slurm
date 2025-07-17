#!/bin/bash

#SBATCH --job-name=OpticalDepth            # Job name
#SBATCH --output=OpticalDepth_%j.out             # Output file name
#SBATCH --error=OpticalDepth_%j.err              # Error file name
#SBATCH --account=cybertrn  			# Account name
#SBATCH --qos=chip-gpu
#SBATCH --cluster=chip-gpu                                              # specify gpu or cpu
#SBATCH --mem=220G                                          # Job memory request 
#SBATCH --constraint="L40S|H100"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=daniellemurphy@berkeley.edu
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=0-02:30:00                                   # Time limit days-hrs:min:sec

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Loading conda environment..."
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate /umbc/rs/nasa-access/users/xingyan/software/miniconda3/envs/satvision

srun python run_cloud_pred.py --config-path configs/16bands.yaml
