#!/bin/bash
#SBATCH -A FUAC6_OHARS
#SBATCH -p m100_all_serial
#SBATCH --time 04:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 8 tasks out of 128
#SBATCH --gres=gpu:1        # 1 gpus per node out of 4
#SBATCH --mem=7600          # memory per node out of 246000MB
#SBATCH --job-name=gx_test_p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /m100/home/userexternal/rjorge00/some_optimizations/GX_SIMSOPT

mpirun -n 1 python3 test_parameters.py
