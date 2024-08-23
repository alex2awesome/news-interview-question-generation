# baseline_type_classification.py

import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluators.vllm_type_classification import classify_question_process_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

if __name__ == "__main__":
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/baseline/QA_Seq_LLM_generated.csv"
    LLM_questions_df = pd.read_csv(dataset_path)
    print(LLM_questions_df)

    type_classified_df = classify_question_process_dataset(LLM_questions_df, output_dir="output_results/baseline", batch_size=40, model_name="meta-llama/Meta-Llama-3-70B-Instruct") # saves type_classification labels in LLM_classified_results.csv
    print(type_classified_df)

    # checked that 8B model works? y/n: y (validated by michael)
