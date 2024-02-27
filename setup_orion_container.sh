#!/bin/bash -xe

DOCKER="sudo docker"
ORION_CTR="orion"
ORION_IMG="fotstrt/orion-ae:v1"
ORION_FORK="orion-fork"
WS=$(git rev-parse --show-toplevel)
DOCKER_WS=$(basename ${WS})

# Setup container
${DOCKER} rm -f ${ORION_CTR} || :
${DOCKER} run -v ${WS}:/root/${DOCKER_WS} -it -d \
    --name ${ORION_CTR} \
    --gpus=all \
    ${ORION_IMG} bash

# Install necessary package
${DOCKER} cp ${ORION_FORK}/setup/nsight-compute.tar ${ORION_CTR}:/usr/local/
${DOCKER} exec -it ${ORION_CTR} bash -c "tar -xf /usr/local/nsight-compute.tar -C /usr/local/"
${DOCKER} exec -it ${ORION_CTR} bash -c "wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_1/nsightsystems-linux-cli-public-2024.1.1.59-3380207.deb"
${DOCKER} exec -it ${ORION_CTR} bash -c "dpkg -i nsightsystems-linux-cli-public-2024.1.1.59-3380207.deb"
${DOCKER} exec -it ${ORION_CTR} bash -c "pip install transformers"
