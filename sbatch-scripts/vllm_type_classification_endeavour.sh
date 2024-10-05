#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=400G
#SBATCH --partition=isi
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=michael.lu@berkeley.edu

cd /project/jonmay_231/spangher/Projects/news-interview-question-generation
source /home1/spangher/.bashrc
conda activate vllm-py310

python -m evaluators.vllm_type_classification
