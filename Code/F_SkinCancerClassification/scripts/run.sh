#!/bin/bash
gpu=0
seeds=(0 1 2 3 4)
method=('fairfed')

for seed in "${seeds[@]}"
do
    for weight in $(seq 0 0.2 1.0)
    do 
        echo "Running ${method} with seed $seed and weight $weight"

        python run.py --seed "$seed" --beta "$weight" --target_GPU "${gpu}" --method ${method} --dp_eo "dp" --sex_age "sex"
        
        if [ $? -ne 0 ]; then
            echo "Error occurred with seed: $seed and weight $weight"
            exit 1
        fi
    done
done

echo "Completed for all seeds."
