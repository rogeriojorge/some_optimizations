#!/bin/bash

#SBATCH -N 9
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=160000 
#SBATCH --time 23:00:00
#SBATCH --job-name=CNT_opt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /marconi_scratch/userexternal/rjorge00/some_optimizations/CNTqs

srun -N 9 ./optimize.py > output.txt
