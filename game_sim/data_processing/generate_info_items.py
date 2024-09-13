# generate_info_items.py

import os
import sys
import re
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from vllm import LLM, SamplingParams
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import load_vllm_model, initialize_tokenizer, stitch_csv_files
from game_sim_prompts import get_info_items_prompt, get_segmented_info_items_prompt

# ---- single use ---- #
def vllm_infer(messages, model, tokenizer):
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    output = model.generate(formatted_prompt, sampling_params)
    return output[0].outputs[0].text

def generate_vllm_response(prompt, role, model, tokenizer):
    messages = [
        {"role": "system", "content": f"{role}."},
        {"role": "user", "content": prompt}
    ]
    return vllm_infer(messages, model, tokenizer)

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
        batch = df.iloc[start_idx:start_idx+batch_size]
        transcripts = batch['combined_dialogue']

        prompts = [get_info_items_prompt(transcript) for transcript in transcripts]
        batch_responses = generate_vllm_info_items_batch(prompts, model, tokenizer)
        
        batch['info_items'] = batch_responses

        batch_file_name = f"batch_{start_idx}_to_{min(start_idx + batch_size, len(df))}_info_item.csv"
        batch_file_path = os.path.join(output_dir, batch_file_name)
        batch.to_csv(batch_file_path, index=False)

    final_df = stitch_csv_files(output_dir, 'final_df_with_info_items.csv')
    return final_df

def clean_info_items(info_items_str):
    return info_items_str.replace('*', '')

def process_info_items(info_items_str):
    items = info_items_str.split('\n')
    info_dict = {}
    current_key = None
    current_value = []

    for item in items:
        item = item.strip()
        match = re.match(r'-?\s*Information item #?(\d+):?\s*(.*)', item)
        if match:
            if current_key:
                info_dict[current_key] = ' '.join(current_value).strip()
            current_key = f"Information item #{match.group(1)}"
            current_value = [match.group(2).strip()]
        elif current_key:
            current_value.append(item)

    if current_key:
        info_dict[current_key] = ' '.join(current_value).strip()

    return info_dict

def extract_segments(response):
    segments = []
    current_segment = ""
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('**Segment'):
            if current_segment:
                segments.append(current_segment.strip())
            current_segment = ""
        elif line and not line.startswith('Here are') and not line.startswith('**'):
            current_segment += line + " "
    if current_segment:
        segments.append(current_segment.strip())
    return segments

def process_segmented_info_items(df, model_name="meta-llama/Meta-Llama-3-70B-Instruct", output_dir="output_results/game_sim/segmented_info_items", batch_size=100):
    os.makedirs(output_dir, exist_ok=True)
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    all_results = []

    for start_idx in range(0, len(df), batch_size):
        batch = df.iloc[start_idx:start_idx+batch_size].copy()
        
        # Clean info items
        batch['info_items'] = batch['info_items'].apply(clean_info_items)
        
        # Process info items for the batch
        batch['info_items_dict'] = batch['info_items'].apply(process_info_items)
        
        all_prompts = []
        for idx, row in batch.iterrows():
            for item_key, item_value in row['info_items_dict'].items():
                prompt = get_segmented_info_items_prompt(row['combined_dialogue'], f"{item_key}: {item_value}")
                all_prompts.append((idx, item_key, prompt))
        
        # Generate segments using LLM
        prompts = [p[2] for p in all_prompts]
        batch_responses = generate_vllm_info_items_batch(prompts, model, tokenizer)
        
        # Process responses
        for (idx, item_key, _), response in zip(all_prompts, batch_responses):
            segments = extract_segments(response)
            if 'segmented_info_items' not in batch.columns:
                batch['segmented_info_items'] = [{}] * len(batch)
            batch.at[idx, 'segmented_info_items'] = batch.at[idx, 'segmented_info_items'].copy()
            batch.at[idx, 'segmented_info_items'][item_key] = segments

        all_results.append(batch)

        # Save batch results
        batch_file_name = f"batch_{start_idx}_to_{min(start_idx + batch_size, len(df))}_segmented_info_items.csv"
        batch_file_path = os.path.join(output_dir, batch_file_name)
        batch.to_csv(batch_file_path, index=False)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(os.path.join(output_dir, 'final_df_with_segmented_info_items.csv'), index=False)
    return final_df

if __name__ == "__main__": 
    # final_dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/dataset/final_dataset.csv"
    # df = pd.read_csv(final_dataset_path)
    # df = df.head(100)
    # print(df)

    # df_with_info_items = process_info_items(df, model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    # print(df_with_info_items)
# ____
    test_file = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/game_sim/outlines/final_df_with_outlines.csv"
    df = pd.read_csv(test_file)
    print(df)

    df_with_segmented_info_items = process_segmented_info_items(df, model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    print(df_with_segmented_info_items)
    print(df_with_segmented_info_items['segmented_info_items'].iloc[0])
