#!/bin/bash
#SBATCH -A FUAC6_OHARS
#SBATCH -p m100_fua_prod
#SBATCH --time 01:59:00     # format: HH:MM:SS
#SBATCH -N 2                # nodes
#SBATCH --ntasks-per-node=4 # tasks out of 128
#SBATCH --gres=gpu:4        # gpus per node out of 4
#SBATCH --mem=100000        # memory per node out of 246000MB
#SBATCH --accel-bind=g
#SBATCH --job-name=gx_opt_mpi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt

cd /m100/home/userexternal/rjorge00/some_optimizations/GX_SIMSOPT

mpirun -n 8 ./main.py
