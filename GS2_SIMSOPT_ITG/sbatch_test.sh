#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=160000 
#SBATCH --time 24:00:00
#SBATCH --job-name=gs2_opt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /marconi_scratch/userexternal/rjorge00/some_optimizations/GS2_SIMSOPT_ITG
mpirun -n 1 ./test_parameters.py
