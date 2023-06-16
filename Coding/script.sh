#!/bin/bash


## Resource Request
#SBATCH --job-name=ECD
#SBATCH --output=ecd_results.out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --gpus-per-task=1 # number of gpus per task
#SBATCH --mem-per-gpu=100M # Memory allocated for the job
#SBATCH --time=0-00:30:00 
## Job Steps

srun echo "Start training with gpu node"
## Displays computing node name
##srun echo "computing node is mentioned below" 
srun echo "*****Computing Node******" 
srun hostname
srun echo "*****Computing Node******" 

##Load the python interpreter
##module load python

source activate monai_ecd
#srun python script.py 90

srun python /homes/ssheikh/postdoc_work/Coding/src/main.py
##srun python script.py 90
srun echo "End training"
