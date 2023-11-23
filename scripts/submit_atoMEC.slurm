#!/bin/bash -l
#SBATCH --job-name=x
#SBATCH --partition=rome
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --mem=50G

module purge
conda activate neuralEOS_cpu
export OMP_NUM_THREADS=1

python -u x.py > x.log
