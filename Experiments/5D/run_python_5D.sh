#!/bin/bash
#SBATCH --job-name=PSO5D                         # Job name
#SBATCH --partition=small                      # Partition name
#SBATCH --array=1-200                           # number of simulations
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=100                           # Number of tasks
#SBATCH --time=00-3:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

common_folder="/home/htc/bzfkostr/GitCodePSO/Unstable-PSO/Experiments/5D"

date;hostname;pwd

cd "$common_folder"
python3 PSO5D.py $SLURM_ARRAY_TASK_ID


date

# run exploration file

#!/bin/bash
#SBATCH --job-name=Exp5D                         # Job name
#SBATCH --partition=big                      # Partition name
#SBATCH --array=1-100                            
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=100                           # Number of tasks
#SBATCH --time=02-1:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

date;hostname;pwd

cd "$common_folder"
python3 ExplorationMultiProcessing5D.py $SLURM_ARRAY_TASK_ID

date

# average exploration
#!/bin/bash
#SBATCH --job-name=AvExp5D                  	# Job name
#SBATCH --partition=big                      # Partition name                   
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=100                           # Number of tasks
#SBATCH --time=02-0:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

date;hostname;pwd

cd "$common_folder"
python3 Exploration5D.py          
date


date

# average the results and calculate mean 
#!/bin/bash
#SBATCH --job-name=Stat5D                  # Job name
#SBATCH --partition=big                      # Partition name                   
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=100                           # Number of tasks
#SBATCH --time=02-0:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

date;hostname;pwd

cd "$common_folder"
python3 Measures5D.py          
date
