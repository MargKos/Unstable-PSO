#!/bin/bash
#SBATCH --job-name=Exp5D                         # Job name
#SBATCH --partition=small                      # Partition name
#SBATCH --array=1-200                            
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --time=00-02:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

date;hostname;pwd

cd "$common_folder"
python3 ExplorationMultiProcessing5D.py $SLURM_ARRAY_TASK_ID



date
