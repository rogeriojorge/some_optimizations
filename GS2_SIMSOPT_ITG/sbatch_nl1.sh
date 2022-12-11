#!/bin/bash

#SBATCH -N 10
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=160000 
#SBATCH --time 24:00:00
#SBATCH --job-name=QA_fin_nl
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /marconi_scratch/userexternal/rjorge00/some_optimizations/GS2_SIMSOPT_ITG
# python3 test_nonlinear.py --type 2
cd nonlinear_nfp2_QA_final_ln1.0_lt3.0
srun -N 10 -n 480 /marconi/home/userexternal/rjorge00/gs2/bin/gs2 gs2Input_ln1.0lt3.0.in > gs2Input_ln1.0lt3.0.log
