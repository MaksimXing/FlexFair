#!/bin/bash
gpu=0
seeds=(0 1 2 3 4)
method='fedavg'

for seed in "${seeds[@]}"
do
    echo "Running ${method} with seed $seed and weight $weight"
    python FedPublicTrain.py --seed "$seed" --method "${method}" --target_GPU "${gpu}"

    if [ $? -ne 0 ]; then
        echo "Error occurred with seed: $seed and weight $weight"
        exit 1
    fi

done

echo "Completed for all seeds."