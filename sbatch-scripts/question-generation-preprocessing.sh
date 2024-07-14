#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100-80gb
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --partition=gpu

source /project/jonmay_231/spangher/Projects/news-interview-question-generation/env_setup.sh
cd /project/jonmay_231/spangher/Projects/news-interview-question-generation/data_preprocessing/