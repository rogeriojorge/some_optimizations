#!/bin/bash
#SBATCH -A FUAC6_OHARS
#SBATCH -p m100_fua_prod
#SBATCH --time 24:00:00      # format: HH:MM:SS
#SBATCH -N 1                 # 1 node
#SBATCH --ntasks-per-node=8  # 8 tasks out of 128
#SBATCH --gres=gpu:1         # 1 gpus per node out of 4
#SBATCH --mem=240000         # memory per node out of 246000MB
#SBATCH --job-name=gx_nl1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /m100/home/userexternal/rjorge00/some_optimizations/GX_SIMSOPT

mpirun -n 1 ./test_final.py --type 1 &
# mpirun -n 1 ./test_final.py --type 2 &
# mpirun -n 1 ./test_final.py --type 3 &
# mpirun -n 1 ./test_final.py --type 4 &
wait
