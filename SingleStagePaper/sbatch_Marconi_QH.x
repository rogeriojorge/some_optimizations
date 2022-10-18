#!/bin/bash

#SBATCH -N 9
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=160000 
#SBATCH --time 1:00:00
#SBATCH --job-name=QH_opt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /marconi/home/userexternal/rjorge00/some_optimizations/SingleStagePaper

srun -N 9 python3 qh.py > output.txt
