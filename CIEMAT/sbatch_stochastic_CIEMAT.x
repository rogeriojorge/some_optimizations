#!/bin/bash
#SBATCH -N 4
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=10000 
#SBATCH --time 00:25:00
#SBATCH --job-name=stochastic_CIEMAT
#SBATCH --mail-type=END
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
srun /marconi/home/userexternal/rjorge00/optimizations/CIEMAT/coil_optimization_simsopt_CIEMAT.py
