#!/bin/bash
#SBATCH --job-name=
#SBATCH --mem=48G
#SBATCH --nodes=1                # num nodes: MUST match .yaml file
#SBATCH --gres=gpu:4             # num gpus per node: MUST match .yaml file AND ntasks-per-node=
#SBATCH --ntasks-per-node=4      # num gpus per node: MUST match .yaml file AND gres=gpu:
#SBATCH --time=23:00:00       # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_6000   # see hpcf website
#SBATCH --error=slurm_output/slurm.err
#SBATCH --output=slurm_output/slurm.out

# variables
run_id=''  # CHANGE THIS
# shouldn't change variables below
config_path='../../config/'
config_path+=${run_id}
config_path+='.yaml'

# run
srun python3 ../../train.py -c $config_path
conda deactivate
echo "conda environment deactivated."

# activate conda env
module load Anaconda3/2024.02-1
source /usr/ebuild/software/emerald/software/Anaconda3/2024.02-1/bin/activate
echo "activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate /umbc/rs/cybertrn/reu2024/team2/envs/ada_main  # choose which environment carefully
echo "conda environment activated."

# debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

