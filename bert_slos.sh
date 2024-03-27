#!/bin/bash -e

models=('bert') 
batches=(2 4 8 16 32)

# Run in closed loop
for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="./run.sh --device-type a100 --device-id 2 --modes mps-uncap --duration 300 --distribution closed ${model}-${model}-${batch}"
        echo "$command"
        eval $command
    done
done

# Run in poisson arrival at load of 0.9
for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="./run.sh --device-type a100 --device-id 2 --modes mps-uncap --duration 300 --distribution poisson --load-start 0.9 --load-end 0.9 ${model}-${model}-${batch}"
        echo "$command"
        eval $command
    done
done
