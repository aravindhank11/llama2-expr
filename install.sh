# Basic packages
sudo apt update
sudo apt-get install -y build-essential python3 python3-venv

# Cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-1
sudo apt install nvtop
