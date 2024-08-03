# batch-LLM-question-generation.py

import sys
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import vllm_infer, vllm_infer_batch, load_vllm_model, extract_text_inside_brackets, extract_text_inside_parentheses, create_QA_Sequence_df_N_qa_pairs
from prompts import CONTEXT_GENERATOR_PROMPT, BASELINE_LLM_QUESTION_PROMPT
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def LLM_question_gen_prompt_loader(prompt, QA_seq):
    new_prompt = prompt.format(QA_Sequence=QA_seq)
    messages = [
        {"role": "system", "content": "You are an expert journalistic interviewer."},
        {"role": "user", "content": new_prompt}
    ]
    return messages

# single-use
def LLM_question_generator(prompt, QA_seq, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    messages = LLM_question_gen_prompt_loader(prompt, QA_seq)
    generated_text = vllm_infer(messages, model_name)
    print(f"generated_text: {generated_text}")
    LLM_question = extract_text_inside_brackets(generated_text)
    motivation = extract_text_inside_parentheses(generated_text)

    return LLM_question, motivation

# for batching
def LLM_question_generator_batch(prompt, QA_seqs, model, tokenizer):
    messages_batch = [LLM_question_gen_prompt_loader(prompt, QA_seq) for QA_seq in QA_seqs]
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    outputs = vllm_infer_batch(formatted_prompts, model)
    LLM_questions = [extract_text_inside_brackets(output) for output in outputs]
    motivations = [extract_text_inside_parentheses(output) for output in outputs]
    return LLM_questions, motivations

# batches QA_Seq data into LLM to predict next question, saves as a csv
def LLM_question_process_dataset(prompt, QA_Seq_df, output_dir="output_results", batch_size=100, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    LLM_question_results = []
    motivation_results = []

    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for start_idx in range(0, len(QA_Seq_df), batch_size):
        batch = QA_Seq_df.iloc[start_idx:start_idx + batch_size]
        QA_seqs = batch['QA_Sequence'].tolist()
        LLM_questions, motivations = LLM_question_generator_batch(prompt, QA_seqs, model, tokenizer)

        LLM_question_results.extend(LLM_questions)
        motivation_results.extend(motivations)

    QA_Seq_df['LLM_Question'] = LLM_question_results
    QA_Seq_df['LLM_Motivation'] = motivation_results

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'QA_Seq_LLM_generated.csv')
    QA_Seq_df.to_csv(output_file_path, index=False)
    return QA_Seq_df

if __name__ == "__main__": 
    dataset_path = os.path.join("dataset/test", "test_dataset_1000.csv")
    df = create_QA_Sequence_df_N_qa_pairs(dataset_path, 3) # saves QA_sequence in QA_Sequence_and_next_question.csv
    LLM_questions_df = LLM_question_process_dataset(BASELINE_LLM_QUESTION_PROMPT, df, model_name="meta-llama/Meta-Llama-3-8B-Instruct") # saves LLM_questions in QA_Seq_LLM_generated.csv
    print(df.head())
