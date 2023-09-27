#!/bin/bash
#SBATCH --job-name=PSO1D                         # Job name
#SBATCH --partition=big                      # Partition name
#SBATCH --array=1-100                            
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=100                           # Number of tasks
#SBATCH --time=00-1:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

common_folder="/home/htc/bzfkostr/GitCodePSO/Unstable-PSO/1D"

date;hostname;pwd

cd "$common_folder"
python3 PSO1D.py $SLURM_ARRAY_TASK_ID


date

#!/bin/bash
#SBATCH --job-name=Stat1D                        # Job name
#SBATCH --partition=big                      # Partition name                   
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=100                           # Number of tasks
#SBATCH --time=02-0:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

date;hostname;pwd

cd "$common_folder"
python3 Measures1D.py            

# run also the Exploration file in 1D
#!/bin/bash
#SBATCH --job-name=Exploration1D                         # Job name
#SBATCH --partition=big                      # Partition name                   
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=100                           # Number of tasks
#SBATCH --time=10-0:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

cd "$common_folder"
python3 Exploration1D.py          

date
