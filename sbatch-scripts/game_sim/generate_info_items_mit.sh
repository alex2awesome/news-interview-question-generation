#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=100
#SBATCH --mem=400G
#SBATCH --partition=sched_mit_psfc_gpu_r8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.lu@berkeley.edu

cd /project/jonmay_231/spangher/Projects/news-interview-question-generation

source env_setup.sh

export OMP_NUM_THREADS=50

python -m game_sim.data_processing.generate_info_items
python -m game_sim.data_processing.generate_segmented_info_items
python -m game_sim.data_processing.generate_outlines
python -m game_sim.conduct_interviews_basic
python -m game_sim.conduct_interviews_intermediate
