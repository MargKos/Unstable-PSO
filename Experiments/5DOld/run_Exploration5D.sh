#!/bin/bash
#SBATCH --job-name=Exp5D                         # Job name
#SBATCH --partition=micro                      # Partition name    
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --time=00-2:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

date;hostname;pwd

cd /home/htc/bzfkostr/GitCodePSO/Unstable-PSO/Experiments/5D
python3 Exploration5D.py 


date
