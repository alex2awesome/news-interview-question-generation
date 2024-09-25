# generate_outlines.py

import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from vllm import LLM, SamplingParams
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import load_vllm_model, initialize_tokenizer, extract_text_inside_brackets
from game_sim_prompts import get_outline_prompt

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

def extract_outlines(transcripts, model, tokenizer):
    information_items = []

    for transcript in transcripts:
        prompt = get_outline_prompt(transcript)
        response = generate_vllm_response(prompt, "You are an AI generating an outline from an interview transcript", model, tokenizer)
        information_items.append(response)

    return information_items

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

def extract_outlines_batch(transcripts, model, tokenizer, batch_size=100):
    outlines = []
    
    for i in range(0, len(transcripts), batch_size):
        batch = transcripts[i:i+batch_size]
        prompts = [get_outline_prompt(transcript) for transcript in batch]
        batch_responses = generate_vllm_outline_batch(prompts, model, tokenizer)
        outline_responses = [extract_text_inside_brackets(response) for response in batch_responses]
        outlines.extend(outline_responses)
    
    return outlines

# ----------- #
def process_outlines(df, model_name="meta-llama/Meta-Llama-3-70B-Instruct", output_dir="output_results/game_sim/outlines"):
    os.makedirs(output_dir, exist_ok=True)
    
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    df['outlines'] = extract_outlines_batch(df['combined_dialogue'], model, tokenizer)
    
    output_file_path = os.path.join(output_dir, "final_df_with_outlines.csv")
    df.to_csv(output_file_path, index=False)
    print(f"csv file saved to {output_file_path}")
    return df

if __name__ == "__main__": 
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/game_sim/segmented_info_items/final_df_with_segmented_info_items.csv"
    df = pd.read_csv(dataset_path)
    print(df)

    df_with_outlines = process_outlines(df, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    print(df_with_outlines)

    # checked that it works with the 8B model? y/n: y 
    # (validated by michael)
