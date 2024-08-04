import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluators.vllm_consistency_eval import consistency_compare_process_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# this prompt instructs LLM to evaluate two different questions based on dimensions of similarity
'''
Dimensions of Similarity:
    1. Informational: Do the questions target the same specific information or facts?
    2. Motivational: Do the questions have the same motivation or underlying purpose?
    3. Contextual: Are both questions equally appropriate for the specific context provided?
    4. Stylistic: Do the questions have similar styles in terms of tone, complexity, and structure?

    Given these dimensions of similarity as well as the following information below, please evaluate whether the two questions below are similar or not. They are either similar or they aren't. The two questions are two possible continuation questions an interviewer can ask given the current interview so far.

    Transcript context: {transcript_context}

    Question 1: {LLM_question}
    Question 1 Type Classification: {LLM_question_type}

    Question 2: {human_question}
    Question 2 Type Classification: {Actual_question_type}

    Please take things step by step. The format of your response should be in this sequence:
    1. First, explain your thought process for each dimension. 
    2. Then, answer the following question: In the context of this interview, are the two questions provided more similar or different? 
    
    Please format your final answer as either similar or different with brackets: [similar] or [different]
    Make sure that only your final answer has brackets.
'''

if __name__ == "__main__":
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/outline/LLM_classified_results.csv"
    type_classified_df = pd.read_csv(dataset_path)
    
    consistency_eval_df = consistency_compare_process_dataset(type_classified_df, output_dir="output_results/outline", model_name="meta-llama/Meta-Llama-3-70B-Instruct") # saves consistency_eval labels in LLM_consistency_eval_results.csv
    print(consistency_eval_df)

    # checked that 8B model works? y/n: y (validated by michael)
