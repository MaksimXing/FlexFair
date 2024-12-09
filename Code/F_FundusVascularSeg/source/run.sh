 #!/bin/bash

seeds=(0 1 2 3 4)
gpu=0
methods=(FairFed)
# weights=(0 0.03 0.06 0.09 0.12 0.15)

for seed in "${seeds[@]}"; do
    for method in "${methods[@]}"; do
        # for weight in "${weights[@]}"; do
        for weight in $(seq 0 0.02 0.10); do
            echo "Running $method with seed $seed and weight ${weight}"
            python run.py $seed $method ${weight} ${gpu}
        done
    done
done

