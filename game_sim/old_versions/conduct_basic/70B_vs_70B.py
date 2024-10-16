import os
import sys
import re
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import find_project_root
from game_sim.conduct_interviews_basic import conduct_basic_interviews_batch
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_path, 'news-interview-question-generation')
    dataset_path = os.path.join(project_root, "output_results/game_sim/outlines/final_df_with_outlines.csv")
    df = pd.read_csv(dataset_path)
    print(df)

    num_turns = 8
    simulated_interviews = conduct_basic_interviews_batch(num_turns, 
                                                          df, 
                                                          model_name="meta-llama/Meta-Llama-3-70B-Instruct",
                                                          output_dir="output_results/game_sim/conducted_interviews_basic/interviewer-70B-vs-source-70B")
    print(simulated_interviews)