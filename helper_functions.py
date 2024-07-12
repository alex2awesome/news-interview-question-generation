# helper_functions.py

import tiktoken
import re
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import os
import torch
import pandas as pd

# ------------- LLM section ------------- #

# hugging face environment setup for VLLM functionality
def setup_hf_env():
    HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
    config_data = json.load(open('/project/jonmay_231/spangher/Projects/news-interview-question-generation/configs/config.json'))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
    os.environ['HF_HOME'] = HF_HOME
    return HF_HOME

# vllm framework model loader
def load_vllm_model(model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    hf_home = setup_hf_env()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    model = LLM(
        model_name,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=hf_home,
        enforce_eager=True
    )
    return model

def vllm_infer(messages, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    output = model.generate(formatted_prompt, sampling_params)
    return output[0].outputs[0].text

# setup openai API
def get_openai_client(key_file_path='~/.openai-api-key.txt'):
    key_path = os.path.expanduser(key_file_path)
    client = OpenAI(api_key=open(key_path).read().strip())
    return client

# ------------- dataset prep section ------------- #

# concats speaker to dialogue, ie. "blablabla", "speaker1" --> "speaker1: blablabla"
def combine_all_qa_pair(row):
    combined = []
    current_speaker = None
    speakers = eval(row['speaker'])
    utterances = eval(row['utt'])
    
    for speaker, utterance in zip(speakers, utterances):
        if speaker != current_speaker:
            combined.append(f"\n{speaker}: {utterance}")
            current_speaker = speaker
        else:
            combined.append(utterance)
    return " ".join(combined)

def combine_two_qa_pairs(row):
    combined = []
    current_speaker = None
    qa_pair_count = 0
    speakers = eval(row['speaker'])
    utterances = eval(row['utt'])
    
    for speaker, utterance in zip(speakers, utterances):
        if "host" in speaker.lower() and speaker != current_speaker:
            qa_pair_count += 1

        if qa_pair_count > 3:
            break

        if speaker != current_speaker:
            combined.append(f"\n{speaker}: {utterance}")
            current_speaker = speaker
        else:
            combined.append(utterance)

    return " ".join(combined)

def create_combined_dialogue_df(dataset_filepath, output_dir="output_results"):
    df = pd.read_csv(dataset_filepath, on_bad_lines='skip')
    df['combined_dialogue'] = df.apply(lambda row: combine_two_qa_pairs(row), axis=1)
    df = df.drop(columns=['utt', 'speaker'])
    
    combined_file_path = os.path.join(output_dir, "combined_data_with_dialogue.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(combined_file_path, index=False)
    return df

# ------------- extract data section ------------- #

# given "ABC[XYZ]EFG", return "XYZ"
def extract_text_inside_brackets(text):
    match = re.search(r'\[(.*?)\]', text)
    if match:
        return match.group(1)
    return 0

# given "ABC{XYZ}EFG", return "XYZ"
def extract_text_inside_parentheses(text):
    match = re.search(r'\((.*?)\)', text)
    if match:
        return match.group(1)
    return 0

# ------------- MISC section ------------- #

def count_tokens(prompts, model='gpt-4o'):
    enc = tiktoken.encoding_for_model(model)
    tok_count = 0
    for p in prompts:
        tok_count += len(enc.encode(p))
    return tok_count

def price_calculator(tok_count, model='gpt-4o', batch=False):
    if batch:
        return f'total price: ${0.0000025 * tok_count}'
    return f'total price: ${0.000005 * tok_count}'