#!/bin/bash
gpu=0
seeds=(0 1 2 3 4)
method='fednova'
weights=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for seed in "${seeds[@]}"
do
    for weight in "${weights[@]}"
    do
        echo "Running ${method} with seed $seed and weight $weight"
        python FedPrivateTrain.py --seed "$seed" --method "${method}" --target_GPU "${gpu}" --rho "${weight}"

        if [ $? -ne 0 ]; then
            echo "Error occurred with seed: $seed and weight $weight"
            exit 1
        fi
    done
done

echo "Completed for all seeds."