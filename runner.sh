#!/bin/bash

# noises=(0.1 0.2 0.3 0.4)
noises=(0.1)
data="TM"
epochs=10
for noise in "${noises[@]}"; do
    echo "Running $noise"
    python3 main.py --priv_budget $noise --data $data --output "output"/$data --epochs $epochs

done