#!/bin/bash

noises=(0.1 0.2 0.3 0.4)
# cr_ratios=(0.1 0.2 0.3 0.4)
cr_ratios=(0.3)
echo "Starting"
datasets=("Cora" "baron_human" "baron_mose" "Segerstolpe" "Zheng" "TM")
# noises=(0.1)
data="Zheng"
epochs=10
central="True"
fed="True"
for data in "${datasets[@]}"; do
    for cr_ratio in "${cr_ratios[@]}"; do
        for noise in "${noises[@]}"; do
            echo "Running $noise"
            python3 main.py --priv_budget $noise --data $data --output "output"/$data --epochs $epochs --central $central --fed $fed --cr_ratio $cr_ratio
        done
    done 
done