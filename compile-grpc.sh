#!/bin/bash

python3 -m grpc_tools.protoc \
    -I /home/ps35324/llama2-expr/protos \
    --python_out=/home/ps35324/llama2-expr/src \
    --grpc_python_out=/home/ps35324/llama2-expr/src \
    /home/ps35324/llama2-expr/protos/*.proto

# sed -i -E 's/^import.*_pb2/from . \0/' /home/$USER/gpu-sharing-scheduler/generated/*.py
