#!/bin/bash

#SBATCH --job-name=PP_LoRA            # Job name
#SBATCH --output=slurm_logs/PhasePred/LoRA/14bands/%j.out             # Output file name
#SBATCH --error=slurm_logs/PhasePred/LoRA/14bands/%j.err              # Error file name
#SBATCH --account=pi_jianwu
#SBATCH --reservation=jianwu                    # Account name
#SBATCH --qos=chip-gpu
#SBATCH --cluster=chip-gpu                                              # specify gpu or cpu
#SBATCH --mem=120G                                          # Job memory request 
#SBATCH --constraint="L40S|H100"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=daniellemurphy@berkeley.edu
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-01:30:00                                   # Time limit days-hrs:min:sec

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Loading conda environment..."
module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate /umbc/rs/nasa-access/users/xingyan/software/miniconda3/envs/satvision
source env_local.sh
echo "$DS_2D_CLOUD_ROOT"
srun python run_cloud_pred.py --config-path configs/Phase_LoRA_14.yaml
