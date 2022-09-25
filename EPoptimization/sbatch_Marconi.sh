#!/bin/bash

#SBATCH -N 1 --ntasks-per-node=36
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=160000 
#SBATCH --time 23:00:00
#SBATCH --job-name=EP_opt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /marconi/home/userexternal/rjorge00/some_optimizations/EPoptimization
export OMP_NUM_THREADS=36
srun --with-pmix -N 1 ./main.py > output.txt
