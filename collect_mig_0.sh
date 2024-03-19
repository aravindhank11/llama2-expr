#!/bin/bash -e

models=('mobilenet_v2' 'mobilenet_v3_small' 'mobilenet_v3_large' 'densenet121' 'densenet161' 'densenet169')
batches=(2 4 8 16 32)

# First run with 4 and 3
for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="./run.sh --device-type a100 --device-id 0 --modes mig --duration 60 --distribution closed vision-${model}-${batch} vision-${model}-${batch}"
        echo "$command"
        eval $command
    done
done

# Second run with 2, 2, 2, 1
for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="./run.sh --device-type a100 --device-id 0 --modes mig --duration 60 --distribution closed vision-${model}-${batch} vision-${model}-${batch} vision-${model}-${batch} vision-${model}-${batch}"
        echo "$command"
        eval $command
    done
done
