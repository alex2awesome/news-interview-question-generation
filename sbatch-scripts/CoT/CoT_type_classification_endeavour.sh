#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=200G
#SBATCH --partition=isi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.lu@berkeley.edu

cd /project/jonmay_231/spangher/Projects/news-interview-question-generation

source /home1/spangher/.bashrc
conda activate vllm-py310

python -m variations.CoT.CoT_type_classification
