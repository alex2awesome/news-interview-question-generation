#!/bin/bash

conda create --name myenv python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate myenv

conda install pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia -y
conda install -c nvidia cudatoolkit-dev=12.2 -y
conda install conda-forge::cudnn=8.1 -y
conda install transformers pandas numpy tqdm -y

pip install -r requirements.txt