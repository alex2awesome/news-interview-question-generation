#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
echo "Script directory: $SCRIPT_DIR"

if conda env list | grep -q 'myenv'; then
    echo "Conda environment 'myenv' already exists. Skipping creation."
    eval "$(conda shell.bash hook)"
    conda activate myenv
else
    conda create --name myenv python=3.10 -y
    eval "$(conda shell.bash hook)"
    conda activate myenv

    conda install pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia -y
    conda install -c nvidia cudatoolkit-dev=12.2 -y
    conda install conda-forge::cudnn=8.1 -y
    conda install transformers pandas numpy=1.26.4 tqdm scikit-learn -y

    pip install -r "$SCRIPT_DIR/requirements.txt"
fi
