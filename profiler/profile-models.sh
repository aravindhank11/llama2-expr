#!/bin/bash

# models=('mobilenet_v2' 'mobilenet_v3_small' 'mobilenet_v3_large' 'densenet121' 'densenet161' 'densenet169' 'densenet201' 'vgg11' 'vgg13' 'vgg16' 'vgg19' 'inception_v3' 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152' 'alexnet' 'squeezenet1_0' 'squeezenet1_1' 'efficientnet_b5' 'efficientnet_b6' 'efficientnet_b7' )
models=('llama' 'bert')
batches=(2 4 8 16 32)

# for model in "${models[@]}"; do
#     for batch in "${batches[@]}"; do
#         command="./tiebreaker-profiler.sh a100 0 $model $batch vision"
#         echo "Profiling ${model} with batch size ${batch}"
#         eval "$command"
#         echo "--------------------------------------------"
#         echo ""
#     done
# done

for model in "${models[@]}"; do
    for batch in "${batches[@]}"; do
        command="python3 extract_profile.py --model ${model} --batch_size ${batch} --profile_dir ./data/a100/${model}/"
        echo "Aggregating profile for ${model} with batch size ${batch}"
        eval "$command"
    done
done
