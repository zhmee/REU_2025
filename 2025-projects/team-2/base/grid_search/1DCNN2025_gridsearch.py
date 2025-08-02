import os
import yaml
import subprocess
import getpass
from sklearn.model_selection import ParameterGrid

#Edit the parameters as needed to grid search, add any additional numbers to the rows as necessary
#Edit the Slurm script time as needed
#Under GRID COMBOS section, certain values are hardcoded and held constant for each run. change as needed.

# ==PARAMETERS=====
#CHANGE THESE
# add what numbers to test for hyperparameters
param_grid = {
    'batch_size': [256],  # change as needed
    'patience' : [1500], #change as needed
    'hidden_layers': [[2048,2048,2048,2048,678,576,512,256,256,256,256,256,256,64,32,16],
        [512, 256, 256, 256, 256, 256, 256, 128]],
    'lr': [0.0005, 0.0001, 0.00005],  # change as needed
    'lr_step':[1000], #change as needed
    'lr_gam':[0.1], #change as needed
    'dropout': [0.05, 0.2, 0.45],  # change numbers and add to rows as necessary
    'data_name': ['shakeri-obe'],  # change dataset names as needed
    'l2':[0.01]
}

# === NEEDED PATHS ===
Template_path = '../../grid_search/1DCNN2025.yaml' #grid_search folder is stored in base
config_output = '../../config'
slurm_out = '../../runs/'

# ===LOAD IN TEMPLATE ====
with open(Template_path) as f:
    template = yaml.safe_load(f)

# ===CREATING RUNID =========
# ===MAKE SURE IT IS THE SAME AS YOUR FOLDER NAME =========
username = ''  # ADD YOUR USERNAME HERE
date_and_time = ''  # ex: 5.3.25 for may 3 2025
run_name = ''  # name whatever this run name is

# ===CHECKING DIRECTORY EXISTENCE ====
os.makedirs(config_output, exist_ok=True)

# ===GRID COMBOS =====
grid = ParameterGrid(param_grid)

for i, params in enumerate(grid):
    tc = template.copy()
    run_id = f"{username}{date_and_time}{run_name}_{i+1}"
    tc['run_id'] = run_id
    tc['pred_ckpt'] = ''
    tc['resume_ckpt'] = ''
    tc['mdl_key'] = 'cnn_1d_2025'
    tc['data']['train_data_path'] = f"/umbc/rs/cybertrn/reu2025/team2/research/base/pp2/data/{params['data_name']}/train/"
    tc['data']['test_data_path'] = f"/umbc/rs/cybertrn/reu2025/team2/research/base/pp2/data/{params['data_name']}/test/"
    tc['data']['batch_size'] = params['batch_size']
    tc['data']['val_split'] =  0.1 #change if needed
    tc['fit']['max_epochs'] = 4000  #change if needed
    tc['fit']['n_nodes'] = 1  # Usually 1
    tc['fit']['n_devices'] = 4  # num of GPUs
    tc['fit']['patience'] = params['patience']  # if needed
    tc['fit']['ckpt_freq'] = 500 #CHANGE IF NEEDED
    tc['model']['input_size'] =  15
    tc['model']['num_classes'] = 13
    tc['model']['hidden_layers'] = params['hidden_layers']
    tc['model']['activation'] = 'relu'
    tc['model']['lr'] = params['lr']
    tc['model']['lr_step'] = params['lr_step']
    tc['model']['lr_gam'] = params['lr_gam']
    tc['model']['penalty'] = 1
    tc['model']['dropout'] = params['dropout']
    tc['model']['l2'] = params['l2']
    # write config YAML
    yaml_path = f"{config_output}/{run_id}.yaml"
    with open(yaml_path, 'w') as f_out:
        yaml.dump(tc, f_out)

    # build SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={run_id}
#SBATCH --cluster=chip-gpu
#SBATCH --mem=48G
#SBATCH --nodes=1                # num nodes: MUST match .yaml file
#SBATCH --gres=gpu:4             # num gpus per node: MUST match .yaml file AND ntasks-per-node=
#SBATCH --ntasks-per-node=4      # num gpus per node: MUST match .yaml file AND gres=gpu:
#SBATCH --time=3-23:00:00          # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_6000    # see hpcf website
#SBATCH --error=slurm_output/slurm.err
#SBATCH --output=slurm_output/slurm.out

# variables
run_id={run_id}
# shouldn't change variables below
config_path='../../config/'${{run_id}}'.yaml'

# activate conda env
module load Anaconda3/2024.02-1
source /usr/ebuild/software/emerald/software/Anaconda3/2024.02-1/bin/activate
echo "activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate /umbc/rs/cybertrn/reu2024/team2/envs/ada_main  # choose which environment carefully
echo "conda environment activated."

# To avoid threading error
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

# debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# run
srun python3 ../../train.py -c $config_path
conda deactivate
echo "conda environment deactivated."
"""

    # check writing out slurm file
    slurm_directory = slurm_out + run_id
    os.makedirs(slurm_directory, exist_ok=True)
    # Give user used hyperparameter combination for the run
    param_path = f"{slurm_directory}/used_hyperparams.yaml"
    with open(param_path, 'w') as param_file:
        yaml.dump(params, param_file)

    slurm_path = f"{slurm_directory}/{run_id}.sh"
    with open(slurm_path, 'w') as the_slurm:
        the_slurm.write(slurm_script)

    # sbatch Job
    subprocess.run(['sbatch', slurm_path], check=True)
