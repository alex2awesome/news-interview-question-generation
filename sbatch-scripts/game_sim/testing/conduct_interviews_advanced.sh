#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=500G
#SBATCH --partition=sched_mit_psfc_gpu_r8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.lu@berkeley.edu

cd /pool001/spangher/alex/news-interview-question-generation
source env_setup.sh

export OMP_NUM_THREADS=50

# test run
python -m game_sim.conduct_interviews_advanced \
    --interviewer_model_name "gpt-4o-mini" \
    --source_model_name "gpt-4o" \
    --batch_size 5 \
    --dataset_path "output_results/game_sim/outlines/final_df_with_outlines.csv" \
    --game_level "advanced"

# NOTE: batch_size is currently set for testing