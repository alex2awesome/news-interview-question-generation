# LLM-question-generation.py

import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import vllm_infer_batch, extract_text_inside_brackets, extract_text_inside_parentheses, create_combined_dialogue_df
from prompts import CONTEXT_GENERATOR_PROMPT, LLM_QUESTION_GENERATOR_PROMPT

def LLM_question_gen_prompt_loader(QA_seq):
    prompt = LLM_QUESTION_GENERATOR_PROMPT.format(QA_Sequence=QA_seq)
    messages = [
        {"role": "system", "content": "You are a world-class interview question guesser."},
        {"role": "user", "content": prompt}
    ]
    return messages

def LLM_question_generator_batch(QA_seqs, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    messages = [LLM_question_gen_prompt_loader(QA_seq) for QA_seq in QA_seqs]
    generated_texts = vllm_infer_batch(messages, model_name)
    LLM_questions = [extract_text_inside_brackets(text) for text in generated_texts]
    motivations = [extract_text_inside_parentheses(text) for text in generated_texts]
    return LLM_questions, motivations

# reformats dataset transcript --> QA_sequence, feeds each QA_Seq into LLM to predict next question, saves prediction
def LLM_question_process_dataset(file_path, output_dir="output_results", batch_size=8):
    dataset = create_combined_dialogue_df(file_path, output_dir)
    question_results = []
    motivation_results = []

    # process dataset in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset.iloc[i:i + batch_size]
        QA_seqs = batch['combined_dialogue'].tolist()
        questions, motivations = LLM_question_generator_batch(QA_seqs, "meta-llama/Meta-Llama-3-8B-Instruct")
        question_results.extend(questions)
        motivation_results.extend(motivations)

    results_df = pd.DataFrame({
        'QA_Sequence': dataset['combined_dialogue'],
        'Generated_Question': question_results,
        'Motivation': motivation_results
    })

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'LLM_generated_results.csv')
    results_df.to_csv(output_file_path, index=False)

if __name__ == "__main__": 
    dataset_path = os.path.join("dataset", "combined_data.csv")
    LLM_question_process_dataset(dataset_path)
    df = pd.read_csv(os.path.join("output_results", "LLM_generated_results.csv"))
    print(df.head())
