#!/bin/bash

elements=("H" "He" "B" "C" "N" "O" "Ne" "Na" "Mg" "Al" "Si")

python database_split.py

for element in "${elements[@]}"; do
    for i in {0..4}; do
        python train_test_split.py $element $i
    done
done    
