# REU 2025 Pytorch Caney Fine-Tuning Documentation!

Authors : Danielle Murphy, Kevin Zhang, Caleb Parten, Autumn Sterling, Haoxiang Zhang

Mentors: Jianwu Wang, RA: Xingyan Li
In collaboration with Jie Gong, Jordan A. Caraballo-Vega, Mark L. Carroll


# Satvision-toa : Fine-Tuning with pytorch-caney
Python package which utilizes Nasa pytorch-caney package to fine tune Satvision-toa model
# Original Documentation :
- Latest : [https://nasa-nccs-hpda.github.io/pytorch-caney/latest](https://nasa-nccs-hpda.github.io/pytorch-caney/latest)
# Objectives :
- Improve performance of existing model in geospatial imaging, using Fine tuning techniques
# Environment Requirements
1. Load in Anaconda
```bash
module load Anaconda3/2024.02-1
```
2. next, activate your virtual environment 
```bash
python -m venv myvenv
source myvenv/bin/activate
```
3. 
``` bash
conda activate myvenv
```
4. You can install all of the requirements like so
``` bash
pip install -r requirements.txt
```

# User Guide
---
# Preprocessing 
---
- TBA
# Running Fine Tuning Satvision-TOA Pipelines
---
## Command-Line Interface (CLI)

- To run tasks using the model, a typical command will look like this :
```bash
python pytorch-caney/pytorch_caney/ptc_cli.py -- config-path <config-path>
```


| Prediction Task       | Fine Tuning Method | Example                                   |
| --------------------- | ------------------ | ----------------------------------------- |
| Fine tuning CloudMask | LoRA               | --config-path configs/CloudMask_LoRA.yaml |
| Fine tuning PhasePred | VPT                | --config-path configs/PhasePred_VPT.yaml  |
| Fine tuning CloudMask | Full Fine Tuning   | --config-path configs/CloudMask_FFT.yaml  |
| ...                   |                    |                                           |
### Examples :
The python file `run_cloud_pred.py` will work with any of the above configurations: 
#### Running CloudMask with LoRA
```
$ python run_cloud_pred.py --config-path configs/CloudMask_LoRA.yaml
```
#### Running Cloud Phase Prediction with VPT
```
$ python run_cloud_pred.py --config-path configs/PhasePred_LoRA.yaml
```
# Cloud Phase Prediction Pipeline
---
This overview will describe how to run the pipelines in this repository, and explain the structure of the codebase. 

*Note : this requires that you download the appropriate Cloud dataset*
##  Pipeline Overview
---
- The operational/functional code lives in `pytorch_caney/`, with the pipelines living in `pytoch_caney/pipelines/`
- For each task, there is a corresponding pipeline:
	- cloud_mask_pipeline.py
	- phase_pred_pipline.py
	- regression_pipeline.py
	- etc...
- Each of these are configurable through YAML files and leverage several customizable variables for:
	- the individual task
	- the model encoder & decoder
	- loss functions
	- metrics
	- and fine-tuning method specific parameters
## Running the Pipelines :
---
- Examples from the original ==pytorch-caney== pre-training are provided in `pytorchcaney/configs/pretraining_unused/` :
> 	-  `3dcloudtask_swinv2_satvision_giant_test.yaml`: Configures a pipeline using the SwinV2-based SatVision encoder.
> 	- `3dcloudtask_fcn_baseline_test.yaml`: Configures a baseline pipeline with a fully convolutional network (FCN).

- We modified these files to better suit the new tasks and dataset.
### Configuration Details :
---
- The specs of the configurable model components are described in  `pytorchcaney/pytorch_caney/configs/config.py`, under the heading "*Model Settings*"
- Below is an example of how you might structure a configuration to finetune the model's Cloud Mask prediction using Visual Prompt Tuning :
```bash
PIPELINE: 'CloudMask'
DATAMODULE: null
MODEL:
  PREPROCESSOR:
    NAME: 'conv' # acceptable is mlp or conv
    HIDDEN_DIMS: [16,16]
    CONV_KERNEL_SIZE: 3
  ENCODER: 'satvision'
  DECODER: 'fcn'
  PRETRAINED: satvision-toa-giant-patch8-window8-128/mp_rank_00_model_states.pt             # OR your pretrained model path
  TYPE: swinv2
  NAME: phasepredict16
  FREEZE_ALL: True                         #Freeze all layers of encoder
  FREEZE_LAYER: 3                          #Freeze up to particular layer
  LORA: False                              #False, VPT
  LORA_RANK: 64 
  VPT: True                                #Activate chosen method
  NUM_PROMPTS: 300                         #Number of prompts to inject
  IN_CHANS: 16                             #Constant, 16 band ABI input
  DROP_PATH_RATE: 0.1
  SWINV2:
    IN_CHANS: 14
    EMBED_DIM: 512
    DEPTHS: [2, 2, 42, 2 ]
    NUM_HEADS: [ 16, 32, 64, 128]
    WINDOW_SIZE: 8
    NORM_PERIOD: 6
DATA:
  BATCH_SIZE: 128 
  DATA_PATHS: []
  TEST_DATA_PATHS: []
  IMG_SIZE: 128 
TRAIN:
  USE_CHECKPOINT: True
  EPOCHS: 30
  WARMUP_EPOCHS: 5
  BASE_LR: 3e-4 
  MIN_LR: 2e-4
  WARMUP_LR: 1e-4
  WEIGHT_DECAY: 0.05
  ACCELERATOR: 'gpu'
  LR_SCHEDULER:
    NAME: 'multistep'
    GAMMA: 0.1
    MULTISTEPS: [700,]
LOSS:
  NAME: 'focal'
PRECISION: "bf16"
PRINT_FREQ: 10 
SAVE_FREQ: 1
VALIDATION_FREQ: 10
TAG: CM_VPT_300                            # Edit as you see fit
```
## Key Components
---
### Model
- **Encoder:** '*Extracts features from input data*', no need to change this
- **Decoder:** *'Processes features in an intermediate representation'*, remains unchanged for now in our project
- **Segmentation Head:** Will '*produce the final output*' with shape `91x40`
**Data & Training:**
- **Batch Size** : customizable parameter, number of training samples used in forward/backward pass per one epoch
- **Epochs** : single iterative pass through the dataset, can be changed, tests were largely done using 25-30
**Loss Function**
- Focal loss (`'focal'`) is used for CloudMask in particular, PhasePrediction was largely tested using cross entropy loss (`'ce'`)
### Notes:
- Customize your `DATAMODULE` as is appropriate

# Putting it all Together :
---
- Our workflow heavily included slurm files to track jobs
- An example of a slurm file consisting of all of the components we've covered so far is below

```bash
#!/bin/bash
#SBATCH --job-name=CM_VPT       # Job name

#SBATCH --output=slurm_logs/CloudMask/VPT/%j.out     # Output file name
#SBATCH --error=slurm_logs/CloudMask/VPT/%j.err      # Error file name
#SBATCH --account=<account-name>                     # Account name
#SBATCH --qos=chip-gpu
#SBATCH --cluster=chip-gpu                           # specify gpu or cpu
#SBATCH --mem=220G                                   # Job memory request 
#SBATCH --constraint="L40S|H100"
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-01:30:00                          # Time limit days-hrs:min:sec

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load Anaconda3/2024.02-1
source ~/.bashrc
conda activate myvenv                               # Name of your env

srun python run_cloud_pred.py --config-path configs/CloudMask_VPT.yaml
```
