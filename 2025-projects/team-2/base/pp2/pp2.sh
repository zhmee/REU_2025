#!/bin/sh
#SBATCH --job-name=pp2
#SBATCH --mem=300G
#SBATCH --nodes=1                # num nodes: MUST match .yaml file
#SBATCH --gres=gpu:1             # num gpus per node: MUST match .yaml file AND ntasks-per-node=
#SBATCH --ntasks-per-node=1      # num gpus per node: MUST match .yaml file AND gres=gpu:
#SBATCH --time=2:00:00       # Time limit days-hrs:min:sec
#SBATCH --error=slurm.err
#SBATCH --output=slurm.out

TAKI=/umbc/rs
ADA=/umbc/rs

###############################
# Edit these
###############################
module load Python/3.12.3-GCCcore-13.3.0   # updated as of 7/8/25
# module load Python/3.10.4-GCCcore-11.3.0-bare  # ada
CLSTR=$TAKI  # $TAKI or $ADA
PROGRAM_BASE=${CLSTR}/cybertrn/reu2025/team2/research/base   # don't change this
PIPELINE_BASE=${PROGRAM_BASE}/pp2/barajas_pp
INPUT_NAME=patient_combined.csv
OUTPUT_NAME=patient_combined
SPLIT=0.05


############################### #
# Run pp2.py in the PIPELINE_BASE
###############################
python ${PIPELINE_BASE}/pp2.py \
    -i ${PROGRAM_BASE}/pp1/data/${INPUT_NAME} \
    -d ${OUTPUT_NAME} \
    -s ${SPLIT} \
