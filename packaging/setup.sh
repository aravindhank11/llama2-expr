#!/bin/bash -xe

# Works on Ubuntu 22 hosts

export DEBIAN_FRONTEND=noninteractive
apt update
apt-get install -y build-essential python3 wget git curl python3-pip vim bc
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cuda-12-1 nvtop
