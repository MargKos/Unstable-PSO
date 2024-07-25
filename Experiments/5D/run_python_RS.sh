#!/bin/bash
#SBATCH --job-name=RS                     # Job name
#SBATCH --partition=big                      # Partition name
#SBATCH --mail-type=END                      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de            # Where to send mail
#SBATCH --nodes=1                            # Run all processes on a single node
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=1                  # Number of CPUs per task
#SBATCH --time=00-5:00:00                   # Time limit (DD-HH:MM:SS)
#SBATCH --output=job_%j.log                  # Standard output and error log

date; hostname; pwd

cd /home/htc/bzfkostr/GitCodePSO/Unstable-PSO/Experiments/5D || exit  # Exit if cd fails
python3 RandomSampling.py

date
