#!/bin/bash
#SBATCH -A FUAC6_OHARS
#SBATCH -p m100_fua_prod
#SBATCH --time 12:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=4 # 8 tasks out of 128
#SBATCH --gres=gpu:1        # 1 gpus per node out of 4
#SBATCH --mem=10000          # memory per node out of 246000MB
#SBATCH --job-name=gx_test_p
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /m100/home/userexternal/rjorge00/some_optimizations/GX_SIMSOPT

mpirun -n 1 ./test_parameters.py
#mpirun -n 1 ./test_parameters.py &
#mpirun -n 1 ./test_parameters.py --nstep 12000 &
#mpirun -n 1 ./test_parameters.py --nstep 12000 --dt 0.175 &
#mpirun -n 1 ./test_parameters.py --nzgrid 121 &
#mpirun -n 1 ./test_parameters.py --nzgrid 121 --npol 6 &
#mpirun -n 1 ./test_parameters.py --nhermite 18 &
#mpirun -n 1 ./test_parameters.py --nlaguerre 8 &
#mpirun -n 1 ./test_parameters.py --nu_hyper 0.25 &
#mpirun -n 1 ./test_parameters.py --D_hyper 0.025 &
#mpirun -n 1 ./test_parameters.py --ny 80 &
#mpirun -n 1 ./test_parameters.py --nx 200 &
#mpirun -n 1 ./test_parameters.py --ny 80 --y0 20.0 &
#wait
