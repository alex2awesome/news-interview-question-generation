#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=150G
#SBATCH --partition=sched_mit_psfc_gpu_r8
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=michael.lu@berkeley.edu

cd /pool001/spangher/alex/news-interview-question-generation

source /home/spangher/.bashrc
conda activate alex

python -m evaluators.vllm_consistency_eval \
    --dataset_path /pool001/spangher/alex/news-interview-question-generation/output_results/outline/QA_Seq_LLM_generated.csv \
    --output_dir /pool001/spangher/alex/news-interview-question-generation/output_results/consistency_eval_outline \
    --verbose \
    --eval_type "multidimensional"


"""
Local run:

python -m evaluators.vllm_consistency_eval \
    --dataset_path output_results/outline/LLM_classified_results.csv \
    --output_dir output_results/consistency_eval_outline \
    --model_name "gpt4-o" \
    --debug \
    --batch_size 1000 \
    --verbose \
    --eval_type "multidimensional"


"""