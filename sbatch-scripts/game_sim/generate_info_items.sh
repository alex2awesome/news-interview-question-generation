#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100-80gb
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.lu@berkeley.edu

cd /project/jonmay_231/spangher/Projects/news-interview-question-generation

source env_setup.sh

python -m game_sim.data_processing.generate_info_items
python -m game_sim.data_processing.generate_segmented_info_items
python -m game_sim.data_processing.generate_outlines
python -m game_sim.conduct_interviews_basic
python -m game_sim.conduct_interviews_intermediate
