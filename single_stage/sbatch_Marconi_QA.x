#!/bin/bash

#SBATCH -N 24
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=160000 
#SBATCH --time 23:00:00
#SBATCH --job-name=QA_opt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /marconi/home/userexternal/rjorge00/optimizations/single_stage

mkdir -p outputs

for ncoils in {3, 4}; do \
    for len in {14,16,18,20,22,24}; do \
        srun -N 1 ./main.py QA --lengthbound $len --stage2 --single_stage --ncoils $ncoils > outputs/QA23_length${len}_${ncoils}coils.txt &
        srun -N 1 ./main.py QA --lengthbound $len --single_stage --ncoils $ncoils > outputs/QA3_length${len}_${ncoils}coils.txt &
        srun -N 1 ./main.py QA --lengthbound $len --stage1 --stage2 --ncoils $ncoils > outputs/QA12_length${len}_${ncoils}coils.txt &
        srun -N 1 ./main.py QA --lengthbound $len --stage1 --stage2 --single_stage --ncoils $ncoils > outputs/QA123_length${len}_${ncoils}coils.txt &
    done
    wait
done
wait
