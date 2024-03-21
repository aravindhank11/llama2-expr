#!/bin/bash -e

models=('mobilenet_v2' 'mobilenet_v3_small' 'mobilenet_v3_large' 'densenet121') 
# models=('densenet161' 'densenet169' 'densenet201' 'vgg11') 
# models=('vgg13' 'vgg16' 'vgg19')
# models=('inception_v3' 'resnet18' 'resnet34') 
# models=('resnet50' 'resnet101' 'resnet152')
# models=('alexnet' 'squeezenet1_0' 'squeezenet1_1')
# models=('efficientnet_b5' 'efficientnet_b6' 'efficientnet_b7')
batches=(2 4 8 16 32)

# Run in closed loop
for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="./run.sh --device-type a100 --device-id 0 --modes mps-uncap --duration 60 --distribution closed vision-${model}-${batch}"
        echo "$command"
        # eval $command
    done
done

# Run in poisson arrival at load of 0.9
for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="./run.sh --device-type a100 --device-id 0 --modes mps-uncap --duration 60 --distribution poisson --load-start 0.9 --load-end 0.9 vision-${model}-${batch}"
        echo "$command"
        # eval $command
    done
done