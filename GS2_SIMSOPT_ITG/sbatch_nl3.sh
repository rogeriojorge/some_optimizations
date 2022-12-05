#!/bin/bash

#SBATCH -N 128
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=160000 
#SBATCH --time 12:00:00
#SBATCH --job-name=QH_fin_nl
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /marconi/home/userexternal/rjorge00/some_optimizations/GS2_SIMSOPT_ITG
# python3 test_nonlinear.py --type 4
cd nonlinear_nfp4_QH_final_ln1.0_lt3.0
srun -N 128 /marconi/home/userexternal/rjorge00/gs2/bin/gs2 gs2Input_ln1.0lt3.0.in > gs2Input_ln1.0lt3.0.log
