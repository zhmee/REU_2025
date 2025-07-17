#!/bin/bash

#SBATCH --job-name=satvision_rerun              # Job name
#SBATCH --output=satvision_chip.out             # Output file name
#SBATCH --error=satvision_chip.err              # Error file name
#SBATCH --account=cybertrn                      # Account name
#SBATCH --cluster=chip-gpu      			    # specify gpu or cpu
#SBATCH --mem=220G                              # Job memory request
#SBATCH --gres=gpu:4                            # Number of requested GPU(s) 
#SBATCH --constraint=L40S                       # Hardware constraint, e.g., L40S for NVIDIA L40S GPU 
#SBATCH --time=0-06:00:00                       # Time limit days-hrs:min:sec

echo "Loading conda environment..."
module load Anaconda3/2024.02-1
source ~/.bashrc 
conda activate /umbc/rs/nasa-access/users/xingyan/software/miniconda3/envs/satvision

python ptc_cli.py --config-path configs/3dcloudtask_swinv2_satvision_giant_test.yaml
