#!/bin/bash

#SBATCH -N 24
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=160000 
#SBATCH --time 23:00:00
#SBATCH --job-name=QH_opt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /marconi/home/userexternal/rjorge00/optimizations/single_stage

mkdir -p outputs

for ncoils in {5, 6}; do \
    for len in {10,11,12,13,14,15}; do \
        srun -N 1 ./main.py QH --lengthbound $len --stage2 --single_stage --ncoils $ncoils > outputs/QH23_length${len}_${ncoils}coils.txt &
        srun -N 1 ./main.py QH --lengthbound $len --single_stage --ncoils $ncoils > outputs/QH3_length${len}_${ncoils}coils.txt &
        srun -N 1 ./main.py QH --lengthbound $len --stage1 --stage2 --ncoils $ncoils > outputs/QH12_length${len}_${ncoils}coils.txt &
        srun -N 1 ./main.py QH --lengthbound $len --stage1 --stage2 --single_stage --ncoils $ncoils > outputs/QH123_length${len}_${ncoils}coils.txt &
    done
    wait
done
wait

