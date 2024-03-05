FROM ubuntu:22.04

# Set the locale
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV RUNNING_IN_DOCKER=1

# install apt packages
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt update && \
    apt-get install -y build-essential python3 wget git curl python3-pip vim bc && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-12-1 nvtop

# install pip packages
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Setup docker image
RUN git clone --recursive https://github.com/aravindhank11/dotfiles.git && \
    cd dotfiles && ./install.sh
