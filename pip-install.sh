#!/bin/bash -xe
python3 -m venv tie-breaker-venv
source tie-breaker-venv/bin/activate
#pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
