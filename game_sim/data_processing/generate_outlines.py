# generate_outlines.py

import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from vllm import LLM, SamplingParams
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import (
    load_vllm_model, 
    initialize_tokenizer, 
    extract_text_inside_brackets, 
    find_project_root, 
    remove_text_before_keyword,
    vllm_infer,
    generate_vllm_response
)
from game_sim_prompts import get_outline_followup_prompt, get_outline_only_prompt

def extract_outlines_followup(transcripts, model, tokenizer):
    outlines = []

    for transcript in transcripts:
        prompt = get_outline_followup_prompt(transcript)
        response = generate_vllm_response(prompt, "You are an AI generating an outline from an interview transcript", model, tokenizer)
        outlines.append(response)

    return outlines

# ---- batch use ---- #
def vllm_infer_batch(messages_batch, model, tokenizer):
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    outputs = model.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

def generate_vllm_outline_batch(prompts, model, tokenizer):
    messages_batch = [
            [
            {"role": "system", "content": "You are an AI generating an outline from an interview transcript."}, 
            {"role": "user", "content": prompt}
            ] 
            for prompt in prompts
        ]
    return vllm_infer_batch(messages_batch, model, tokenizer)

def generate_vllm_outline_only_batch(outlines, model, tokenizer):
    messages_batch = [
            [
            {"role": "system", "content": "You are an AI specializing in data extraction."}, 
            {"role": "user", "content": prompt}
            ] 
            for prompt in outlines
        ]
    return vllm_infer_batch(messages_batch, model, tokenizer)

def extract_outlines_followup_batch(transcripts, model, tokenizer, batch_size=100):
    outlines = []
    
    for i in range(0, len(transcripts), batch_size):
        batch = transcripts[i:i+batch_size]
        prompts = [get_outline_followup_prompt(transcript) for transcript in batch]
        batch_responses = generate_vllm_outline_batch(prompts, model, tokenizer)
        outline_responses = [extract_text_inside_brackets(response) for response in batch_responses]
        outlines.extend(outline_responses)
    
    return outlines

def extract_outlines_only_batch(outlines, model, tokenizer, batch_size=100):
    objectives = []
    
    for i in range(0, len(outlines), batch_size):
        batch = outlines[i:i+batch_size]
        prompts = [get_outline_only_prompt(outline) for outline in batch]
        batch_responses = generate_vllm_outline_only_batch(prompts, model, tokenizer)
        objectives.extend(batch_responses)
    
    return objectives

# ----------- #
def process_outlines(df, model_name="meta-llama/Meta-Llama-3-70B-Instruct", output_dir="output_results/game_sim/outlines"):
    os.makedirs(output_dir, exist_ok=True)
    
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    print("Generating outlines with follow-ups...")
    df['outlines_with_followups'] = extract_outlines_followup_batch(df['combined_dialogue'], model, tokenizer)

    print("Extracting only objectives from outlines...")
    df['outlines'] = extract_outlines_only_batch(df['outlines_with_followups'], model, tokenizer)
    df['outlines'] = df['outlines'].apply(remove_text_before_keyword)
    
    output_file_path = os.path.join(output_dir, "final_df_with_outlines.csv")
    df.to_csv(output_file_path, index=False)
    print(f"csv file saved to {output_file_path}")
    return df

if __name__ == "__main__": 
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_path, 'news-interview-question-generation')
    dataset_path = os.path.join(project_root, "output_results/game_sim/info_items_dict/final_df_with_info_items_dict.csv")
    df = pd.read_csv(dataset_path)
    print(df)

    df_with_outlines = process_outlines(df, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    print(df_with_outlines)

    # checked that it works with the 8B model? y/n: y 
    # (validated by michael)
