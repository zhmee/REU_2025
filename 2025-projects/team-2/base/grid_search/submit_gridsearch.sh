#!/bin/sh
#SBATCH --job-name=submit_gridsearch
#SBATCH --mem=300G
#SBATCH --nodes=1                # num nodes: MUST match .yaml file
#SBATCH --gres=gpu:1             # num gpus per node: MUST match .yaml file AND ntasks-per-node=
#SBATCH --ntasks-per-node=1      # num gpus per node: MUST match .yaml file AND gres=gpu:
#SBATCH --time=1:00:00       # Time limit days-hrs:min:sec
#SBATCH --error=slurm_submit_gridsearch.err
#SBATCH --output=slurm_submit_gridsearch.out

module load Python/3.12.3-GCCcore-13.3.0   # updated as of 7/8/25

python 1DCNN2025_gridsearch.py
