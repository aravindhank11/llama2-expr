#!/bin/bash -e

models=('densenet201' 'vgg11' 'vgg13' 'vgg16' 'vgg19' 'inception_v3')
batches=(2 4 8 16 32)

# First run with 4 and 3
for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="./run.sh --device-type a100 --device-id 1 --modes mig --duration 60 --distribution closed vision-${model}-${batch} vision-${model}-${batch}"
        echo "$command"
        eval $command
    done
done

# Second run with 2, 2, 2, 1
for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="./run.sh --device-type a100 --device-id 1 --modes mig --duration 60 --distribution closed vision-${model}-${batch} vision-${model}-${batch} vision-${model}-${batch} vision-${model}-${batch}"
        echo "$command"
        eval $command
    done
done