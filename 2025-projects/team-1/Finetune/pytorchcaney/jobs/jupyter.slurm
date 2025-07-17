#!/bin/bash
#SBATCH --job-name=Notebook
#SBATCH --output=jupyter-log-%J.txt
#SBATCH --error=jupyter-log-%J.err
#SBATCH --account=cybertrn
#SBATCH --cluster=chip-cpu
#SBATCH --partition=match
#SBATCH --qos=shared
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --tasks-per-node=1
#SBATCH --mem=64G

## get tunneling info
module load Anaconda3
source activate /umbc/rs/nasa-access/users/xingyan/software/miniconda3/envs/satvision

XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)
jobnum=$(%J)

## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport $USER@chip.rs.umbc.edu
    ------------------------------------------------------------------
    Then go to the .err file associated with this SLURM job and copy the
    line starting with \"http://127.0.0.1\" into your browser.

    To find it easily, try:
        cat <filename>.err | grep 127.0.0.1
    "
python -m pip install jupyter_collaboration
## start an ipcluster instance and launch jupyter server
jupyter-lab --no-browser --port=$ipnport --ip=$ipnip --collaborative
