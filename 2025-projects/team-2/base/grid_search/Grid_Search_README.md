USING GRID SEARCH
ACCESSING BRIDE:
cd research
cd base
cd config

ACCESSING GRIDSEARCH SCRIPTS:
/umbc/rs/cybertrn/reu2025/team2/research/base/grid_search
SETUP:
pip install conda
pip install scikit-learn
pip install pyyaml
**do not launch an interactive session or any conda environment

OR if you do not want to pip install libraries so that you don't overload your home directory, use the submit_gridsearch.sh shell
script (further information below)


TRAINING
Cd runs
Mkdir (run_id) (usernameDataRunName, EX: mzhao7.8.2025test_Grid)
Copy over the associated gridsearch file for the model
For 1D CNN :
cp ../../grid_search/1DCNN2025_gridsearch.py .
cp ../../grid_search/submit_gridsearch.sh
Follow documentation instructions to edit hyperparameters and update runname in the py file
Edit allowed Batch time

IF you did pip install:
'Python 1DCNN2025_gridsearch.py' (or whichever gridsearch model you used)

IF you didn't do pip install:
Edit the filename in the submit_gridsearch.sh file that is now in your run directory
sbatch submit_gridsearch.sh

watch squeue -u $USER

SEE OUTPUT INFO
Cd logs
Cd csv_logs
Cd runname_1 (different trailing numbers represent different hyperparamters, look at yaml file for hyperparamters)
Cd lightninglogs
Cd version0
Cat metrics.csv

ADDITIONAL INFO
Each run of GridSearch will generate a new folder that contains information about the hyperparameters that were used for that run.
Within the version0 folder containing the metrics.csv, a yaml file will also be generated with the hyperparamters used for deciding
future runs.
