#!/bin/bash -e

./run.sh --device-type a100 --device-id 0 --modes mig --duration 60 --distribution poisson --load-start 0.2 --load-end 0.2 vision-mobilenet_v3_large-4 vision-mobilenet_v3_large-4 vision-mobilenet_v3_large-4 vision-mobilenet_v3_large-4
./run.sh --device-type a100 --device-id 0 --modes mig --duration 60 --distribution poisson --load-start 0.5 --load-end 0.5 vision-mobilenet_v3_large-16 vision-mobilenet_v3_large-16 vision-mobilenet_v3_large-16 vision-mobilenet_v3_large-16
# ./run.sh --device-type a100 --device-id 1 --modes mig --duration 60 --distribution poisson --load-start 0.2 --load-end 0.2 vision-resnet34-8 vision-resnet34-8 vision-resnet34-8 vision-resnet34-8
# ./run.sh --device-type a100 --device-id 1 --modes mig --duration 60 --distribution poisson --load-start 0.8 --load-end 0.8 vision-resnet34-16 vision-resnet34-16 vision-resnet34-16 vision-resnet34-16
# ./run.sh --device-type a100 --device-id 2 --modes mig --duration 60 --distribution poisson --load-start 0.7 --load-end 0.7 vision-resnet50-16 vision-resnet50-16 vision-resnet50-16 vision-resnet50-16
# ./run.sh --device-type a100 --device-id 2 --modes mig --duration 60 --distribution poisson --load-start 0.6 --load-end 0.6 vision-resnet101-2 vision-resnet101-2 vision-resnet101-2 vision-resnet101-2
# ./run.sh --device-type a100 --device-id 3 --modes mig --duration 60 --distribution poisson --load-start 1.0 --load-end 1.0 vision-resnet101-2 vision-resnet101-2 vision-resnet101-2 vision-resnet101-2
# ./run.sh --device-type a100 --device-id 3 --modes mig --duration 60 --distribution poisson --load-start 1.0 --load-end 1.0 vision-resnet152-16 vision-resnet152-16 vision-resnet152-16 vision-resnet152-16
# ./run.sh --device-type a100 --device-id 3 --modes mig --duration 60 --distribution poisson --load-start 0.2 --load-end 0.2 vision-efficientnet_b6-16 vision-efficientnet_b6-16 vision-efficientnet_b6-16 vision-efficientnet_b6-16

# ./run.sh --device-type a100 --device-id 0 --modes mig --duration 60 --distribution poisson --load-start 0.1 --load-end 1.0 vision-efficientnet_b7-32 vision-efficientnet_b7-32 vision-efficientnet_b7-32 vision-efficientnet_b7-32

