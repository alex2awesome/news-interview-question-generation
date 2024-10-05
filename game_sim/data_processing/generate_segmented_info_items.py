# generate_segmented_info_items.py

import os
import sys
import json
import re
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from vllm import LLM, SamplingParams
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import load_vllm_model, initialize_tokenizer, stitch_csv_files, find_project_root
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
            {"role": "system", "content": "You are an AI assistant helping to segment key information items from an interview transcript."},
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
        
        batch['info_items'] = batch['info_items'].apply(clean_info_items)
        batch['info_items_dict'] = batch['info_items'].apply(process_info_items)
        
        all_prompts = []
        for idx, row in batch.iterrows():
            for item_key, item_value in row['info_items_dict'].items():
                prompt = get_segmented_info_items_prompt(row['combined_dialogue'], f"{item_key}: {item_value}")
                all_prompts.append((idx, item_key, prompt))
        
        prompts = [p[2] for p in all_prompts]
        batch_responses = generate_vllm_info_items_batch(prompts, model, tokenizer)
        
        for (idx, item_key, _), response in zip(all_prompts, batch_responses):
            segments = extract_segments(response)
            if 'segmented_info_items' not in batch.columns:
                batch['segmented_info_items'] = [{}] * len(batch)
            batch.at[idx, 'segmented_info_items'] = batch.at[idx, 'segmented_info_items'].copy()
            batch.at[idx, 'segmented_info_items'][item_key] = segments

        all_results.append(batch)

        batch_file_name = f"batch_{start_idx}_to_{min(start_idx + batch_size, len(df))}_segmented_info_items.csv"
        batch_file_path = os.path.join(output_dir, batch_file_name)
        batch.to_csv(batch_file_path, index=False)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(os.path.join(output_dir, 'final_df_with_segmented_info_items.csv'), index=False)
    return final_df

# def process_segmented_info_items(df, model_name="meta-llama/Meta-Llama-3-70B-Instruct",
#                                  output_dir="output_results/game_sim/segmented_info_items",
#                                  batch_size=100):
#     os.makedirs(output_dir, exist_ok=True)
#     model = load_vllm_model(model_name)
#     tokenizer = initialize_tokenizer(model_name)

#     all_results = []

#     for start_idx in range(0, len(df), batch_size):
#         batch = df.iloc[start_idx:start_idx + batch_size].copy()

#         # Process info items
#         batch['info_items_dict'] = batch['info_items'].apply(process_info_items)

#         all_prompts = []
#         for idx, row in batch.iterrows():
#             for item_key, item_value in row['info_items_dict'].items():
#                 prompt = get_segmented_info_items_prompt(row['combined_dialogue'], f"{item_key}: {item_value}")
#                 all_prompts.append((idx, item_key, prompt))

#         # Generate responses from the LLM
#         prompts = [p[2] for p in all_prompts]
#         batch_responses = generate_vllm_info_items_batch(prompts, model, tokenizer)

#         # Initialize 'segmented_info_items' column if not present
#         if 'segmented_info_items' not in batch.columns:
#             batch['segmented_info_items'] = [{} for _ in range(len(batch))]

#         # Process responses
#         for (idx, item_key, _), response in zip(all_prompts, batch_responses):
#             segments = extract_segments(response)
#             # Add segments to the corresponding row and item
#             batch.at[idx, 'segmented_info_items'][item_key] = segments

#         # Serialize 'segmented_info_items' as JSON strings before saving
#         batch['segmented_info_items'] = batch['segmented_info_items'].apply(json.dumps)

#         # Save the batch
#         batch_file_name = f"batch_{start_idx}_to_{min(start_idx + batch_size, len(df))}_segmented_info_items.csv"
#         batch_file_path = os.path.join(output_dir, batch_file_name)
#         batch.to_csv(batch_file_path, index=False)

#         all_results.append(batch)

#     # Combine all batches
#     final_df = pd.concat(all_results, ignore_index=True)

#     # Save the final DataFrame
#     final_df.to_csv(os.path.join(output_dir, 'final_df_with_segmented_info_items.csv'), index=False)

#     return final_df

# def clean_info_items(info_items_str):
#     cleaned_str = re.sub(r'^[\*\-\•\s]+', '', info_items_str, flags=re.MULTILINE)
#     return cleaned_str.strip()

# def extract_segments(response):
#     segments = {}
#     pattern = re.compile(r'Segment\s*(\d+):\s*Title:\s*(.*?)\s*Content:\s*(.*?)(?=(Segment\s*\d+:|$))', re.DOTALL | re.IGNORECASE)
#     matches = pattern.finditer(response)

#     for match in matches:
#         segment_number = match.group(1).strip()
#         title = match.group(2).strip()
#         content = match.group(3).strip()
#         segments[f"Segment {segment_number}"] = {
#             'Title': title,
#             'Content': content
#         }

#     return segments

# def process_info_items(info_items_str):
#     info_items_str = info_items_str.strip()
#     pattern = re.compile(r'-?\s*Information\s*item\s*#?(\d+):?\s*(.*)', re.IGNORECASE)
    
#     info_dict = {}
#     current_key = None
#     current_value = ''

#     lines = info_items_str.split('\n')
#     for line in lines:
#         line = line.strip()
#         match = pattern.match(line)
#         if match:
#             if current_key:
#                 info_dict[current_key] = current_value.strip()
#             item_number = match.group(1)
#             content = match.group(2)
#             current_key = f"Information item #{item_number}"
#             current_value = content.strip()
#         else:
#             if current_key:
#                 current_value += ' ' + line.strip()

#     # Add the last item
#     if current_key:
#         info_dict[current_key] = current_value.strip()

#     return info_dict

if __name__ == "__main__": 
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_path, 'news-interview-question-generation')
    dataset_path = os.path.join(project_root, "output_results/game_sim/info_items/final_df_with_info_items.csv")
    df = pd.read_csv(dataset_path)
    print(df)

    df['info_items'] = df['info_items'].apply(clean_info_items)

    df_with_segmented_info_items = process_segmented_info_items(df, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    print(df_with_segmented_info_items)
    