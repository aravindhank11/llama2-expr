#!/bin/bash -xe

source ../helper.sh

# Login to docker
${DOCKER} login

# build image
${DOCKER} buildx build --platform linux/amd64 -t ${TIE_BREAKER_IMG} .

# push image
${DOCKER} push ${TIE_BREAKER_IMG}
