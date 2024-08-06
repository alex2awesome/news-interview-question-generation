import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import create_QA_Sequence_df_N_qa_pairs # temp import
from LLM_question_generation import LLM_question_process_dataset
from evaluators.vllm_type_classification import classify_question_process_dataset
from evaluators.vllm_consistency_eval import consistency_compare_process_dataset
from prompts import BASELINE_LLM_QUESTION_PROMPT

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# currently testing functionality for baseline.py
'''
    1. from task dataset, use the downsmapled test set --> this dataset has a varied amount of QA pairs
    2. generate LLM questions given the first k QA pairs
    3. generate LLM labels classifying the type of both the LLM question (prediction) and actual question (ground truth) based on the taxonomy
    4. generate evaluation on consistency between the LLM question and actual question
'''

if __name__ == "__main__":
    # test_dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/dataset/test/test_dataset_1000.csv"
    # df = create_QA_Sequence_df_N_qa_pairs(test_dataset_path, 3, output_dir="output_results/test")
    # print(df)
    
    # LLM_questions_df = LLM_question_process_dataset(BASELINE_LLM_QUESTION_PROMPT, df, output_dir="output_results/test", model_name="meta-llama/Meta-Llama-3-8B-Instruct") # saves LLM_questions in QA_Seq_LLM_generated.csv
    # print(LLM_questions_df)
    
    # type_classified_df = classify_question_process_dataset(LLM_questions_df, output_dir="output_results/test", model_name="meta-llama/Meta-Llama-3-8B-Instruct") # saves type_classification labels in LLM_classified_results.csv
    # print(type_classified_df)
    
    # consistency_eval_df = consistency_compare_process_dataset(type_classified_df, output_dir="output_results/test", model_name="meta-llama/Meta-Llama-3-8B-Instruct") # saves consistency_eval labels in LLM_consistency_eval_results.csv
    # print(consistency_eval_df)

    path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/dataset/test/transcripts_with_split_outlines_1000.csv"
    df = pd.read_csv(path)
    print(df.columns)

    