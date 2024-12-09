#!/bin/bash
gpu=0
seeds=(0 1 2 3 4)
method='fairfed'

for seed in "${seeds[@]}"
do
    for weight in $(seq 0 0.02 0.10)
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

