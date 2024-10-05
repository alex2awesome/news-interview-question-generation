# run_all_mit.py

import os
import sys
import re
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.generate_info_items import process_info_items
from data_processing.generate_segmented_info_items import process_segmented_info_items
from data_processing.generate_outlines import process_outlines
from conduct_interviews_basic import conduct_basic_interviews_batch
from conduct_interviews_intermediate import conduct_intermediate_interviews_batch

if __name__ == "__main__": 
    # generate_info_items
    dataset_path = "/pool001/spangher/alex/news-interview-question-generation/output_results/game_sim/info_items/final_df_with_info_items.csv"
    df_with_info_items = pd.read_csv(dataset_path)
    print(df_with_info_items)

    # generate_segmented_info_items

    df_with_segmented_info_items = process_segmented_info_items(df_with_info_items, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    print(df_with_segmented_info_items)

    # generate_outlines

    df_with_outlines = process_outlines(df_with_segmented_info_items, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    print(df_with_outlines)

    # basic setup
    num_turns = 8
    simulated_interviews = conduct_basic_interviews_batch(num_turns, df_with_outlines, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    print(simulated_interviews)

    # intermediate setup
    simulated_interviews = conduct_intermediate_interviews_batch(num_turns, df_with_outlines, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    print(simulated_interviews)
