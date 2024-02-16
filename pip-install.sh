#!/bin/bash -xe
python3 -m venv tie-breaker-venv
source tie-breaker-venv/bin/activate
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
