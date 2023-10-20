#!/usr/bin/env bash

# installing python and pythorch
sudo apt update
sudo apt install python3
pip install torch==2.0.1+rocm5.4.2 torchvision==0.15.2+rocm5.4.2 --index-url https://download.pytorch.org/whl/rocm5.4.2