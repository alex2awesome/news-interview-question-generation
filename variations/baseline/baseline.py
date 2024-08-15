import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LLM_question_generation import LLM_question_process_dataset
from evaluators.vllm_type_classification import classify_question_process_dataset
from evaluators.vllm_consistency_eval import consistency_compare_process_dataset
from prompts import BASELINE_LLM_QUESTION_PROMPT

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# (QASeq only) baseline variation: motivation is asked afterwards so that it doesn't affect the question generated (!= CoT)
'''
Here's the dialogue so far between the interviewer, and the guest (source):

{QA_Sequence}

Please guess the next question the interviewer will ask. Format your final guess for the question in brackets like this: [Guessed Question]. 
Next, please explain the motivation behind the question you provided in paragraph form, then format it with parentheses like this: (motivation explanation)
'''

if __name__ == "__main__":
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/dataset/task_dataset/task_dataset_test.csv"
    df = pd.read_csv(dataset_path)
    df.rename(columns={"Input (first k qa_pairs)": "QA_Sequence", "Ground Truth (k+1 question)": "Actual_Question"}, inplace=True)
    print(df)
    
    LLM_questions_df = LLM_question_process_dataset(BASELINE_LLM_QUESTION_PROMPT, df, output_dir="output_results/baseline", model_name="meta-llama/Meta-Llama-3-70B-Instruct") # saves LLM_questions in QA_Seq_LLM_generated.csv
    print(LLM_questions_df)
    
    type_classified_df = classify_question_process_dataset(LLM_questions_df, output_dir="output_results/baseline", model_name="meta-llama/Meta-Llama-3-70B-Instruct") # saves type_classification labels in LLM_classified_results.csv
    print(type_classified_df)
    
    consistency_eval_df = consistency_compare_process_dataset(type_classified_df, output_dir="output_results/baseline", model_name="meta-llama/Meta-Llama-3-70B-Instruct") # saves consistency_eval labels in LLM_consistency_eval_results.csv
    print(consistency_eval_df)

    '''
    1. from task dataset, use the downsmapled test set --> this dataset has a varied amount of QA pairs
    2. generate LLM questions given the first k QA pairs
    3. generate LLM labels classifying the type of both the LLM question (prediction) and actual question (ground truth) based on the taxonomy
    4. generate evaluation on consistency between the LLM question and actual question
    '''