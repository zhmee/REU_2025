#!/bin/bash

#SBATCH --job-name=satvision_rerun                          # Job name
#SBATCH --output=satvision_chip-%j-%u-slurm.out             # Output file name
#SBATCH --error=satvision_chip-%j-%u-slurm.err              # Error file name
#SBATCH --account=cybertrn                                  # Account name
#SBATCH --cluster=chip-gpu      			                # specify gpu or cpu
#SBATCH --mem=220G                                          # Job memory request
#SBATCH --constraint=L40S
#SBATCH --nodes=2                                           # Number of requested node(s) 
#SBATCH --gpus-per-node=4                                   # Number of requested GPU(s) 
#SBATCH --ntasks-per-node=4
#SBATCH --time=0-01:00:00                                   # Time limit days-hrs:min:sec

echo "Loading conda environment..."
module load Anaconda3/2024.02-1
source ~/.bashrc 
conda activate /umbc/rs/nasa-access/users/xingyan/software/miniconda3/envs/satvision

srun python ptc_cli_n2.py --config-path configs/3dcloudtask_swinv2_satvision_giant_test.yaml