#!/bin/bash
 
#Submit this script with: sbatch run_center.sh
#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=96   # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=2G   # memory per CPU core
#SBATCH -J "bomex_test"   # job name
#SBATCH --mail-user=kyle@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
 
 
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
srun python -u main.py Bomex.in
