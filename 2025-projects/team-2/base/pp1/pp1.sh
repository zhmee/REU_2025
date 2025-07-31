#!/bin/sh
#SBATCH --job-name=pp1_al_patient25
#SBATCH --mem=300G
#SBATCH --nodes=1                # num nodes: MUST match .yaml file
#SBATCH --gres=gpu:1             # num gpus per node: MUST match .yaml file AND ntasks-per-node=
#SBATCH --ntasks-per-node=1      # num gpus per node: MUST match .yaml file AND gres=gpu:
#SBATCH --time=5:00:00       # Time limit days-hrs:min:sec
#SBATCH --error=slurm.err
#SBATCH --output=slurm.out

module load Python/3.12.3-GCCcore-13.3.0  # taki, not sure if old
# conda init bash  # hacky debugging fix, may  need to call a source command
# module load Anaconda3/2021.05
# source /usr/ebuild/software/Anaconda3/2021.05/bin/activate
# conda activate /umbc/xfs1/cybertrn/reu2024/team2/envs/taki_main
# module load Python/3.10.4-GCCcore-11.3.0-bare  # ada
TAKI=/umbc/rs
ADA=/umbc/rs
CLSTR=$TAKI  # change depending on the cluster
PROGRAM_BASE=${CLSTR}/cybertrn/reu2025/team2/research/base
PIPELINE_BASE=${PROGRAM_BASE}/pp1/barajas_pp   # this is where pipeline files can be called from
FINAL_OUTPUT=patient_combined # name of final output in base/pp1/data/
OUTPUT_BASE=${PIPELINE_BASE}/intermediate_output/${FINAL_OUTPUT}  # for now, we hard code this and use this convention

# the below is where the raw data comes from
TO_CONVERT_EVENTS=(
    ${PROGRAM_BASE}/raw_data/al_patient_2025/*_triples.txt
    ${PROGRAM_BASE}/raw_data/patient_2025/*_triples.txt
    ${PROGRAM_BASE}/raw_data/shakeri-obe/*_triples.txt
    ${PROGRAM_BASE}/raw_data/sharma/*_triples.txt
)
TO_CONVERT_CLASSES=(
    ${PROGRAM_BASE}/raw_data/al_patient_2025/*_tType.txt
    ${PROGRAM_BASE}/raw_data/patient_2025/*_tType.txt
    ${PROGRAM_BASE}/raw_data/shakeri-obe/*_tType.txt
    ${PROGRAM_BASE}/raw_data/sharma/*_tType.txt
)

###############################
# Convert all of Polf's provided data into our labelled csv format
###############################
CONVERT_PREFIX=${OUTPUT_BASE}/converted/

mkdir -p ${CONVERT_PREFIX}
python ${PIPELINE_BASE}/convert_polf.py \
    -f \
    -t triples \
    -p ${CONVERT_PREFIX} \
    -i ${TO_CONVERT_EVENTS[@]} \
    -c ${TO_CONVERT_CLASSES[@]}

###############################
# Merge all of the converted data
###############################
MERGE_PREFIX=${OUTPUT_BASE}/merged/

TO_MERGE_TRIPLES=${CONVERT_PREFIX}/*_triples_converted.csv
TO_MERGE_DTOT=${CONVERT_PREFIX}/*_dtot_converted.csv
TO_MERGE_F_TRIPLES=${CONVERT_PREFIX}/*_false_converted.csv

JSON_RATIO='
{
    "triples":                1,
    "dtot":                   1,
    "false_triples":        1/6
}
'
mkdir -p ${MERGE_PREFIX}
python ${PIPELINE_BASE}/data_merger.py \
    --random \
    --merge-preferences none \
    -j "${JSON_RATIO}" \
    -p ${MERGE_PREFIX} \
    --triples ${TO_MERGE_TRIPLES[@]} \
    --dtot ${TO_MERGE_DTOT[@]} \
    --false-triples ${TO_MERGE_F_TRIPLES[@]}
# -> This will produce only a single file since we do not use
# --merge-preferences scatters

# We will run this again but with a preferences for scatters and as is to get all merged files across all kMU and MeV
python ${PIPELINE_BASE}/data_merger.py \
    --as-is \
    --merge-preferences scatters \
    -p ${MERGE_PREFIX} \
    --triples ${TO_MERGE_TRIPLES[@]} \
    --dtot ${TO_MERGE_DTOT[@]} \
    --false-triples ${TO_MERGE_F_TRIPLES[@]}

###############################
# Shuffle the data
###############################
SHUFFLE_PREFIX=${OUTPUT_BASE}/merged_shuffled/

mkdir -p ${SHUFFLE_PREFIX}
python ${PIPELINE_BASE}/shuffler.py \
    -i ${MERGE_PREFIX}/*.csv \
    -p ${SHUFFLE_PREFIX}

###############################
# Make the labels continuous, specific case of 6 and 7 not being used
###############################
python ${PIPELINE_BASE}/to_cont.py \
    -i ${SHUFFLE_PREFIX}/*triples_false_triples_dtot*.csv \
    -o ${PROGRAM_BASE}/pp1/data/${FINAL_OUTPUT}.csv
# Since the shuffler only produced one file we can use * and try to catch the only file in the directory with globbing.
# If there were more than 1 file this would need to be ran in a loop
