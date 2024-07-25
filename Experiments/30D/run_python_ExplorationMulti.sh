#!/bin/bash
#SBATCH --job-name=Exp30D                         # Job name
#SBATCH --partition=big                        # Partition name  
#SBATCH --array=1-100                        
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=1                             # Number of tasks
#SBATCH --time=02-00:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

date;hostname;pwd


cd "$common_folder"
python3 ExplorationMultiProcessing.py $SLURM_ARRAY_TASK_ID