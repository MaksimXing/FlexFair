#!/bin/bash
gpu=0
seeds=(0 1 2 3 4)
method='fedprox'
weights=(0.01 0.02 0.05 0.1 0.5)

for seed in "${seeds[@]}"
do
    for weight in "${weights[@]}"
    do
        echo "Running ${method} with seed $seed and weight $weight"
        python FedPublicTrain.py --seed "$seed" --method "${method}" --target_GPU "${gpu}" --miu "${weight}"

        if [ $? -ne 0 ]; then
            echo "Error occurred with seed: $seed and weight $weight"
            exit 1
        fi
    done
done

echo "Completed for all seeds."