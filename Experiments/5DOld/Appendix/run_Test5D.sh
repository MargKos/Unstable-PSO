#!/bin/bash
#SBATCH --job-name=Test                         # Job name
#SBATCH --partition=big                      	# Partition name    
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=100                           # Number of tasks
#SBATCH --time=02-1:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

date;hostname;pwd

cd /home/htc/bzfkostr/GitCodePSO/Unstable-PSO/5D
python3 Exploration5D.py 

date
