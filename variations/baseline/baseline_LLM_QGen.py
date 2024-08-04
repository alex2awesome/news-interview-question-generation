import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LLM_question_generation import LLM_question_process_dataset
from prompts import BASELINE_LLM_QUESTION_PROMPT

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# (QASeq only) baseline variation: motivation is asked afterwards so that it doesn't affect the question generated (!= CoT)
'''
Your task is to predict the next question that will follow in an interview.
Make sure that you are recognizing the interviewee's last comment and acknowledging it when appropriate, rather than immediately moving on and asking a question. When you do decide acknowledgment is necessary, make sure your response is personal and empathetic (sound like you care about what the interviewee has to say). This can simply be acknowledging what they said.

The format of your response should be in this sequence:
1. First, guess the next question asked by the interviewer. Format your final guess for the question in brackets like this: [Guessed Question]. 
2. Then, explain the main motivation/intent behind the question that should be asked, then format your explanation with parentheses like this: (motivation explanation)

Here is the interview so far:
{QA_Sequence}

Remember to format your guess for the next question the interviewer asks in brackets [], then your motivation explanation in parentheses ().
'''

if __name__ == "__main__":
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/dataset/task_dataset/task_dataset_test.csv"
    df = pd.read_csv(dataset_path)
    df.rename(columns={"Input (first k qa_pairs)": "QA_Sequence", "Ground Truth (k+1 question)": "Actual_Question"}, inplace=True)
    
    LLM_questions_df = LLM_question_process_dataset(BASELINE_LLM_QUESTION_PROMPT, df, output_dir="output_results/baseline", model_name="meta-llama/Meta-Llama-3-8B-Instruct") # saves LLM_questions in QA_Seq_LLM_generated.csv
    print(LLM_questions_df)

    '''
    1. from task dataset, use the downsmapled test set --> this dataset has a varied amount of QA pairs
    2. generate LLM questions given the first k QA pairs
    '''

    # checked that 8B model works? y/n: y (validated by michael)