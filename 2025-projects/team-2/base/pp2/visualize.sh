#!/bin/sh
#SBATCH --job-name=visualize
#SBATCH --mem=300G
#SBATCH --nodes=1                # num nodes: MUST match .yaml file
#SBATCH --gres=gpu:1             # num gpus per node: MUST match .yaml file AND ntasks-per-node=
#SBATCH --ntasks-per-node=1      # num gpus per node: MUST match .yaml file AND gres=gpu:
#SBATCH --time=2:00:00       # Time limit days-hrs:min:sec
#SBATCH --error=Visualize_slurm.err
#SBATCH --output=Visualize_slurm.out

###############################
# Edit these
###############################
module load Python/3.12.3-GCCcore-13.3.0   # updated as of 7/8/25

python visualize.py --path patient_2025/
