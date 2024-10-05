# vllm-type-classification.py

import sys
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from transformers import AutoTokenizer
import gc
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import vllm_infer, vllm_infer_batch, load_vllm_model, extract_text_inside_brackets, stitch_csv_files
from prompts import get_classify_taxonomy_prompt, TAXONOMY
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def type_classification_prompt_loader(QA_seq, question):
    prompt = get_classify_taxonomy_prompt(QA_seq, question)
    messages = [
        {"role": "system", "content": "You are a world-class annotator for interview questions."},
        {"role": "user", "content": prompt}
    ]
    return messages

# single-use
def classify_question(QA_Sequence, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    messages = type_classification_prompt_loader(QA_Sequence)
    generated_text = vllm_infer(messages, model_name)
    print(f"generated_text: {generated_text}")
    
    question_type = extract_text_inside_brackets(generated_text)
          
    if question_type in TAXONOMY:
        return question_type
    else:
        return "Unknown question type"

# for batching
def classify_question_batch(QA_Sequences, questions, model, tokenizer):
    messages_batch = [type_classification_prompt_loader(QA_seq, question) for QA_seq, question in zip(QA_Sequences, questions)]
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    outputs = vllm_infer_batch(formatted_prompts, model)
    question_types = [extract_text_inside_brackets(output) if extract_text_inside_brackets(output).lower() in TAXONOMY else f"{extract_text_inside_brackets(output)}" for output in outputs]
    return question_types

# this adds a column to LLM_questions_df called LLM_Question_Type and Actual_Question_Type
def classify_question_process_dataset(LLM_questions_df, output_dir="output_results", batch_size=50, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    LLM_question_types_results = []
    Actual_question_types_results = []

    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for start_idx in range(0, len(LLM_questions_df), batch_size):
        batch = LLM_questions_df.iloc[start_idx:start_idx + batch_size]
        
        QA_Sequences = batch['QA_Sequence']
        LLM_questions = batch['LLM_Question']
        Actual_questions = batch['Actual_Question']

        LLM_question_types = classify_question_batch(QA_Sequences, LLM_questions, model, tokenizer)
        Actual_question_types = classify_question_batch(QA_Sequences, Actual_questions, model, tokenizer)

        LLM_question_types_results.extend(LLM_question_types)
        Actual_question_types_results.extend(Actual_question_types)

        gc.collect()

    LLM_questions_df['LLM_Question_Type'] = LLM_question_types_results
    LLM_questions_df['Actual_Question_Type'] = Actual_question_types_results

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'LLM_classified_results.csv')
    LLM_questions_df.to_csv(output_file_path, index=False)
    print(f"csv file saved to {output_file_path}")
    return LLM_questions_df

# implementation 2: save by batch + functionality to start where u stop
def efficient_classify_question_process_dataset(LLM_questions_df, output_dir="output_results", batch_size=50, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    os.makedirs(output_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(output_dir) if re.match(r'LLM_classified_results_\d+_\d+\.csv', f)]
    if existing_files:
        last_file = sorted(existing_files, key=lambda x: int(re.search(r'_(\d+)\.csv', x).group(1)))[-1]
        last_end_idx = int(re.search(r'_(\d+)\.csv', last_file).group(1))
        current_idx = last_end_idx
        print(f"Resuming from index {current_idx}")
    else:
        current_idx = 0
    
    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for start_idx in range(current_idx, len(LLM_questions_df), batch_size):
        batch = LLM_questions_df.iloc[start_idx:start_idx + batch_size]
        
        QA_Sequences = batch['QA_Sequence']
        LLM_questions = batch['LLM_Question']
        Actual_questions = batch['Actual_Question']

        LLM_question_types = classify_question_batch(QA_Sequences, LLM_questions, model, tokenizer)
        Actual_question_types = classify_question_batch(QA_Sequences, Actual_questions, model, tokenizer)

        temp_df = batch.copy()
        temp_df['LLM_Question_Type'] = LLM_question_types
        temp_df['Actual_Question_Type'] = Actual_question_types

        output_file_path = os.path.join(output_dir, f'LLM_classified_results_{start_idx}_{start_idx + batch_size}.csv')
        temp_df.to_csv(output_file_path, index=False)
        print(f"Batch {start_idx} to {start_idx + batch_size} saved to {output_file_path}")

        gc.collect()

    print("All batches processed and saved.")
    LLM_classified_df = stitch_csv_files(output_dir, 'final_LLM_classified_results.csv')
    return LLM_classified_df


# classify only human questions: idea is that human ground truth only need to be classified once
def classify_human_question_process_dataset(df, batch_size=50, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    Actual_question_types_results = []

    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for start_idx in range(0, len(df), batch_size):
        batch = df.iloc[start_idx:start_idx + batch_size]
        
        QA_Sequences = batch['QA_Sequence']
        Actual_questions = batch['Actual_Question']

        Actual_question_types = classify_question_batch(QA_Sequences, Actual_questions, model, tokenizer)
        Actual_question_types_results.extend(Actual_question_types)

        gc.collect()
    return Actual_question_types_results

def human_label_appender(df, labels, input_file_path="output_results"):
    df['Actual_Question_Type'] = labels
    
    output_dir = os.path.dirname(input_file_path)
    output_file_path = os.path.join(output_dir, 'LLM_human_question_labels_classified.csv')
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file_path, index=False)
    print(f"csv file saved to {output_file_path}")
    return df

# classify only LLM questions:
def classify_LLM_question_process_dataset(LLM_questions_df, output_dir="output_results", batch_size=50, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    LLM_question_types_results = []

    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for start_idx in range(0, len(LLM_questions_df), batch_size):
        batch = LLM_questions_df.iloc[start_idx:start_idx + batch_size]
        
        QA_Sequences = batch['QA_Sequence']
        LLM_questions = batch['LLM_Question']

        LLM_question_types = classify_question_batch(QA_Sequences, LLM_questions, model, tokenizer)
        LLM_question_types_results.extend(LLM_question_types)

        gc.collect()

    LLM_questions_df['LLM_Question_Type'] = LLM_question_types_results
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'LLM_classified_results.csv')
    LLM_questions_df.to_csv(output_file_path, index=False)
    print(f"csv file saved to {output_file_path}")
    return LLM_questions_df

# function for testing
def is_actual_question_same_for_all_files(file_paths, column_name="Actual_Question"):
    first_df = pd.read_csv(file_paths[0])
    first_column = first_df[column_name].tolist()
    for path in file_paths[1:]:
        df = pd.read_csv(path)
        current_column = df[column_name].tolist()
        
        if first_column != current_column:
            print(f"Mismatch found in file: {path}")
            return False
    
    print("The 'Actual_Question' column is consistent across all files.")
    return True


if __name__ == "__main__":
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/baseline/QA_Seq_LLM_generated.csv"
    df = pd.read_csv(dataset_path)
    df = df.head(10)
    # human_question_type_labels = classify_human_question_process_dataset(df, batch_size=100, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    # path_lst = ["/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/baseline/QA_Seq_LLM_generated.csv", 
    #             "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/CoT/QA_Seq_LLM_generated.csv",
    #             "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/CoT_outline/QA_Seq_LLM_generated.csv",
    #             "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/outline/QA_Seq_LLM_generated.csv"]
    
    # for path in path_lst:
    #     df = pd.read_csv(path)
    #     human_label_appender(df, human_question_type_labels, input_file_path=path)
    human_question_type_labels = classify_human_question_process_dataset(df, batch_size=100, model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    print(human_question_type_labels)
    