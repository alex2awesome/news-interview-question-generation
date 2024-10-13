#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=150G
#SBATCH --partition=isi
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=michael.lu@berkeley.edu

cd /project/jonmay_231/spangher/Projects/news-interview-question-generation

source /home1/spangher/.bashrc
conda activate vllm-retriv-py39


python -m evaluators.vllm_consistency_eval \
    --dataset_path /project/jonmay_1426/spangher/news-interview-question-generation/output_results/CoT/QA_Seq_LLM_generated.csv \
    --output_dir /project/jonmay_1426/spangher/news-interview-question-generation/output_results/consistency_eval_cot

