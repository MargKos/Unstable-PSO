#!/bin/bash
#SBATCH --job-name=PSO30D                        # Job name
#SBATCH --partition=big                          # Partition name  
#SBATCH --array=1-100                        
#SBATCH --mail-type=END                          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de                # Where to send mail
#SBATCH --nodes=1                                # Run all processes on a single node
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --time=02-00:00:00                       # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                   # Standard output and error log


common_folder="/home/htc/bzfkostr/GitCodePSO/Unstable-PSO/30D"
date;hostname;pwd


cd "$common_folder"
python3 PSO.py $SLURM_ARRAY_TASK_ID


date

#!/bin/bash
#SBATCH --job-name=PSO30D_I                     # Job name
#SBATCH --partition=big                         # Partition name  
#SBATCH --array=1-100                        
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de               # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --time=02-00:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

date;hostname;pwd

cd "$common_folder"
python3 PSO_information.py $SLURM_ARRAY_TASK_ID


date

#!/bin/bash
#SBATCH --job-name=30Stat                       # Job name
#SBATCH --partition=big                         # Partition name  
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de               # Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --time=02-00:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

date;hostname;pwd


cd "$common_folder"
python3 Statistics.py 

date

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


date

#!/bin/bash
#SBATCH --job-name=30AvExp                      # Job name
#SBATCH --partition=big                         # Partition name                 
#SBATCH --mail-type=END                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kostre@zib.de              	# Where to send mail
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --ntasks=1                             	# Number of tasks
#SBATCH --time=02-00:00:00                      # Time limit (necessary for Z1)
#SBATCH --output=job_%a-%a.log                  # Standard output and error log

date;hostname;pwd


cd "$common_folder"
python3 Exploration.py

date


