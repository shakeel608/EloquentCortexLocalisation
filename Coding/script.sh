#!/bin/bash


## Resource Request
#SBATCH --job-name=ECD
#SBATCH --output=ecd_results.out
#SBATCH --time=1-00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500M

## Job Steps

srun echo "Start process"
## Displays computing node name
srun echo "computing node is below" 
srun hostname

##Load the python interpreter
##module load python

#srun python script.py 90

srun python main.py

srun echo "End process"
