#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --partition=sched_mit_psfc_gpu_r8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.lu@berkeley.edu

cd /pool001/spangher/alex/news-interview-question-generation
source env_setup.sh

export OMP_NUM_THREADS=16
os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = '3600'
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
nvidia-smi

python -m game_sim.conduct_basic.8B_vs_70B