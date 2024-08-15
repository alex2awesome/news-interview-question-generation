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
from prompts import TAXONOMY, CLASSIFY_USING_TAXONOMY_PROMPT
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def type_classification_prompt_loader(QA_seq, question):
    prompt = CLASSIFY_USING_TAXONOMY_PROMPT.format(transcript_section=QA_seq, question=question)
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
    question_types = [extract_text_inside_brackets(output) if extract_text_inside_brackets(output).lower() in TAXONOMY else f"(MISC) {extract_text_inside_brackets(output)}" for output in outputs]
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

if __name__ == "__main__":
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/test/QA_Seq_LLM_generated.csv"
    df = pd.read_csv(dataset_path)
    new_df = classify_question_process_dataset(df, output_dir="output_results/test/type_classification", batch_size=100, model_name="meta-llama/Meta-Llama-3-8B-Instruct") # saves type_classification labels in LLM_classified_results.csv
    print(new_df)

    # dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/test/type_classification/LLM_classified_results.csv"
    # df = pd.read_csv(dataset_path)
    filtered_df = df[(df["LLM_Question_Type"].str.contains("(MISC)", na=False)) | 
                     (df["Actual_Question_Type"].str.contains("(MISC)", na=False))
                    ]
    LLM_Question_Type = filtered_df["LLM_Question_Type"].tolist()
    Actual_Question_Type = filtered_df["Actual_Question_Type"].tolist()
    count = 0
    for guess, actual in zip(LLM_Question_Type, Actual_Question_Type):
        if "(MISC)" in guess:
            count += 1
            print(guess)
        if "(MISC)" in actual:
            count += 1
            print(actual)
    print(f"proportion of the errors in a sample of 300 data points: {count/df.shape[0]}")
 
    # new_df = classify_question_process_dataset(df, output_dir="output_results/test/type_classification", model_name="meta-llama/Meta-Llama-3-8B-Instruct") # saves type_classification labels in LLM_classified_results.csv
    # print(new_df)

    # expected result: dataframe now contains the following columns: QA_Sequence, Actual_Question, LLM_Question, LLM_Question_Type, Actual_Question_Type
