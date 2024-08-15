import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LLM_question_generation import OUTLINE_LLM_question_process_dataset
from prompts import OUTLINE_LLM_QUESTION_PROMPT
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# (QASeq + Outline) variation: additional context is provided along with the QA Sequence
'''
Your task is to predict the next question that will follow in an interview. I will give you the current interview dialogue as well as the motivation behind the interview.
Make sure that you are recognizing the interviewee's last comment and acknowledging it when appropriate, rather than immediately moving on and asking a question. When you do decide acknowledgment is necessary, make sure your response is personal and empathetic (sound like you care about what the interviewee has to say). This can simply be acknowledging what they said.

The format of your response should be in this sequence:
1. First, guess the next question asked by the interviewer. Format your final guess for the question in brackets like this: [Guessed Question]. 
2. Then, explain the main motivation/intent behind the question that should be asked, then format your explanation with parentheses like this: (motivation explanation)

Here is the relevant information:
{outline_statement}

Here is an outline of your goals and top questions you want to ask for the interview:
{interview_goals}

{general_questions}

Here is the interview so far:
{QA_Sequence}

Remember to format your guess for the next question asked in brackets [], then your motivation in parentheses ().
'''

if __name__ == "__main__":
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/dataset/task_dataset/task_dataset_test.csv"
    df = pd.read_csv(dataset_path)
    df.rename(columns={"Input (first k qa_pairs)": "QA_Sequence", "Ground Truth (k+1 question)": "Actual_Question"}, inplace=True)
    
    LLM_questions_df = OUTLINE_LLM_question_process_dataset(OUTLINE_LLM_QUESTION_PROMPT, df, output_dir="output_results/outline", model_name="meta-llama/Meta-Llama-3-70B-Instruct") # saves LLM_questions in QA_Seq_LLM_generated.csv
    print(LLM_questions_df)

    '''
    1. from task dataset, use the downsampled test set --> this dataset has a varied amount of QA pairs
    2. generate LLM questions given the first k QA pairs and an outline
    '''

    # checked that 8B model works? y/n: y (validated by michael)