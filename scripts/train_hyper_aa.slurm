#!/bin/bash -l
#SBATCH --partition=casus    # partition to be used (defq, gpu or intel)
#SBATCH -A casus
#SBATCH --time=48:00:00       # walltime (up to 96 hours)
#SBATCH --nodes=1            # number of nodes
#SBATCH --ntasks=10         # number of tasks (i.e. parallel processes) to be started

export LC_ALL='en_US.utf8'
module purge
module load gcc
module load openmpi
module load cuda/11.8

conda activate neuralEOS_gpu

python -u ~/neuralEOS/scripts/train_inner_template.py $1 $2 $3 $4 $5 $SLURM_JOB_ID
