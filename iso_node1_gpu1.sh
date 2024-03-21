#!/bin/bash -e

models=('densenet161' 'densenet169' 'densenet201' 'vgg11') 
batches=(2 4 8 16 32)

# Run in closed loop
for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="./run.sh --device-type a100 --device-id 1 --modes mps-uncap --duration 60 --distribution closed vision-${model}-${batch}"
        echo "$command"
        eval $command
    done
done

# Run in poisson arrival at load of 0.9
for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="./run.sh --device-type a100 --device-id 1 --modes mps-uncap --duration 60 --distribution poisson --load-start 0.9 --load-end 0.9 vision-${model}-${batch}"
        echo "$command"
        eval $command
    done
done