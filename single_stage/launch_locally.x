#!/bin/bash

cd /Users/rogeriojorge/local/optimizations/single_stage

mkdir -p outputs

for len in {10,11,12,13}; do \
    ./main.py QH --lengthbound $len --stage2 --single_stage > outputs/QH23_$len.txt
    ./main.py QH --lengthbound $len --single_stage > outputs/QH3_$len.txt
    ./main.py QH --lengthbound $len --stage1 --stage2 > outputs/QH12_$len.txt
    ./main.py QH --lengthbound $len --stage1 --stage2 --single_stage > outputs/QH123_$len.txt
done

for len in {18,20,22,24}; do \
    ./main.py QA --lengthbound $len --stage2 --single_stage > outputs/QA23_$len.txt
    ./main.py QA --lengthbound $len --single_stage > outputs/QA3_$len.txt
    ./main.py QA --lengthbound $len --stage1 --stage2 > outputs/QA12_$len.txt
    ./main.py QA --lengthbound $len --stage1 --stage2 --single_stage > outputs/QA123_$len.txt
done

# len=20
# ./main.py QA --lengthbound $len --stage2 --single_stage > outputs/QA23_$len.txt
# ./main.py QA --lengthbound $len --single_stage > outputs/QA3_$len.txt
# ./main.py QA --lengthbound $len --stage1 --stage2 --single_stage > outputs/QA123_$len.txt
# ./main.py QA --lengthbound $len --stage1 --stage2 > outputs/QA12_$len.txt

# len=11
# ./main.py QH --lengthbound $len --stage2 --single_stage > outputs/QH23_$len.txt
# ./main.py QH --lengthbound $len --single_stage > outputs/QH3_$len.txt
# ./main.py QH --lengthbound $len --stage1 --stage2 --single_stage > outputs/QH123_$len.txt
# ./main.py QH --lengthbound $len --stage1 --stage2 > outputs/QH12_$len.txt