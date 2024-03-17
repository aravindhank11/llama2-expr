#!/bin/bash -xe

source ../helper.sh

# Login to docker
${DOCKER} login

# build image
${DOCKER} build -t ${TIE_BREAKER_IMG} .

# push image
${DOCKER} push ${TIE_BREAKER_IMG}
