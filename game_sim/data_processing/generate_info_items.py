# generate_info_items.py

import os
import sys
import re
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from vllm import LLM, SamplingParams
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import (
    load_vllm_model, 
    initialize_tokenizer, 
    stitch_csv_files, 
    find_project_root,
    generate_vllm_response

)
from game_sim_prompts import get_info_items_prompt


def extract_information_items(transcripts, model, tokenizer):
    information_items = []

    for transcript in transcripts:
        prompt = get_info_items_prompt(transcript)
        response = generate_vllm_response(prompt, "You are an AI extracting key information items from an interview transcript", model, tokenizer)
        information_items.append(response)

    return information_items

# ---- batch use ---- #
def vllm_infer_batch(messages_batch, model, tokenizer):
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    outputs = model.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

def generate_vllm_info_items_batch(prompts, model, tokenizer):
    messages_batch = [
            [
            {"role": "system", "content": "You are an AI extracting key information items from an interview transcript"}, 
            {"role": "user", "content": prompt}
            ] 
            for prompt in prompts
        ]
    return vllm_infer_batch(messages_batch, model, tokenizer)

def extract_information_items_batch(transcripts, model, tokenizer, batch_size=100):
    information_items = []
    
    for i in range(0, len(transcripts), batch_size):
        batch = transcripts[i:i+batch_size]
        prompts = [get_info_items_prompt(transcript) for transcript in batch]
        batch_responses = generate_vllm_info_items_batch(prompts, model, tokenizer)
        information_items.extend(batch_responses)
    
    return information_items

# ----------- #
def process_info_items(df, model_name="meta-llama/Meta-Llama-3-70B-Instruct", output_dir="output_results/game_sim/info_items", batch_size=100):
    os.makedirs(output_dir, exist_ok=True)
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    for start_idx in range(0, len(df), batch_size):
        batch = df.iloc[start_idx:start_idx+batch_size].copy()
        transcripts = batch['combined_dialogue']

        prompts = [get_info_items_prompt(transcript) for transcript in transcripts]
        batch_responses = generate_vllm_info_items_batch(prompts, model, tokenizer)
        
        batch['info_items'] = batch_responses

        batch_file_name = f"batch_{start_idx}_to_{min(start_idx + batch_size, len(df))}_info_item.csv"
        batch_file_path = os.path.join(output_dir, batch_file_name)
        batch.to_csv(batch_file_path, index=False)

    final_df = stitch_csv_files(output_dir, 'final_df_with_info_items.csv')
    return final_df

if __name__ == "__main__": 
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_path, 'news-interview-question-generation')
    dataset_path = os.path.join(project_root, "dataset/final_dataset.csv")
    df = pd.read_csv(dataset_path)
    sampled_df = df.sample(n=500, random_state=42)
    print(sampled_df)

    df_with_info_items = process_info_items(sampled_df, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    print(df_with_info_items)
