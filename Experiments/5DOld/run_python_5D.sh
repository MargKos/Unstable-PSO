#!/bin/bash
#SBATCH --job-name=PSO5D                         # Job name
#SBATCH --partition=small                      # Partition name
#SBATCH --array=1-200                           # number of simulations
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=100                           # Number of tasks
#SBATCH --time=00-0:02:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

common_folder="/home/htc/bzfkostr/GitCodePSO/Unstable-PSO/Experiments/5D"

date;hostname;pwd

cd "$common_folder"
python3 PSO5D.py $SLURM_ARRAY_TASK_ID
