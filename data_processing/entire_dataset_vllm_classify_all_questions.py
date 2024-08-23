import sys
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import combine_csv_files
from vllm_classify_all_questions import classify_each_question
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

if __name__ == "__main__":
    df = pd.read_csv("/project/jonmay_231/spangher/Projects/news-interview-question-generation/dataset/final_dataset.csv")
    print(df)
    interviewNum = len(df)
    classify_each_question(df, num_interviews=interviewNum, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    directory_path = 'output_results/vllm_all_questions_classified'
    output_file = 'output_results/vllm_all_questions_classified/entire_dataset_classified_v1.csv'
    combine_csv_files(directory_path, output_file)
    