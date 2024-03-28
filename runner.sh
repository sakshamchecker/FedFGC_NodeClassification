#!/bin/bash

# noises=(0.1 0.2 0.3 0.4)
noises=(0.1)
# cr_ratios=(0.1 0.2 0.3 0.4)
cr_ratios=(0.3)
echo "Starting"
# datasets=("Cora" "Citeseer" "PubMed" "baron_human" "baron_mose" "Segerstolpe" "Zheng" "TM" )
datasets=("Cora")

#Model Define
epochs=30
# models=('GCN' 'GAT' 'Sage')
models=('GCN')

#Setting
central="True"
fed="False"
#DP privacy
privacy='False'
#Coarsening
coarsen='All'
#------------------------
for data in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for cr_ratio in "${cr_ratios[@]}"; do
            for noise in "${noises[@]}"; do
                echo "Running $data $model $cr_ratio $noise"
                python3 main.py --priv_budget $noise --data $data --model $model --output "output"/$data --privacy $privacy --coarsen $coarsen --epochs $epochs --central $central --fed $fed --cr_ratio $cr_ratio
            done
        done 
    done
done