#!/bin/bash

conda update -n base conda -y
conda create -n Transformer_torch python=3.11.9 anaconda -y
source activate Transformer_torch

pip install -r /root/Transformer_pytorch/requirements.txt

git config --global user.name "knlpscience"
git config --global user.email knlpscientist@gmail.com