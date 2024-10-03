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
import torch.distributed as dist


# ------------- LLM section ------------- #

# hugging face environment setup for VLLM functionality
def setup_hf_env():
    dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
    path_to_config = os.path.join(dir_of_this_script, 'configs', 'config.json')
    with open(path_to_config, 'r') as config_file:
        config_data = json.load(config_file)
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
    return os.getenv('HF_HOME')

# vllm framework model loader
def load_vllm_model(model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    hf_home = setup_hf_env()
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

    model = LLM(
        model_name,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=hf_home,
        enforce_eager=True
    )

    memory_allocated = torch.cuda.memory_allocated()
    memory_reserved = torch.cuda.memory_reserved()
    
    print(f"Model {model_name} loaded. Memory Allocated: {memory_allocated / (1024 ** 3):.2f} GB")
    print(f"Model {model_name} loaded. Memory Reserved: {memory_reserved / (1024 ** 3):.2f} GB")
    return model

def initialize_tokenizer(model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

# for batching
def vllm_infer_batch(messages_batch, model):
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    outputs = model.generate(messages_batch, sampling_params)
    return [output.outputs[0].text for output in outputs]

# for single-use testing only
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

def combine_N_qa_pairs_and_next_question(row, N):
    combined = []
    current_speaker = None
    qa_pair_count = 0
    next_question = None
    speakers = eval(row['speaker'])
    utterances = eval(row['utt'])
    last_host_question = None
    
    for speaker, utterance in zip(speakers, utterances):
        if "host" in speaker.lower() and speaker != current_speaker:
            last_host_question = f"{speaker}: {utterance}"
            qa_pair_count += 1

        if qa_pair_count > N:
            if "host" in speaker.lower():
                next_question = f"{speaker}: {utterance}"
            break

        if speaker != current_speaker:
            combined.append(f"\n{speaker}: {utterance}")
            current_speaker = speaker
        else:
            combined.append(utterance)

    if next_question is None and last_host_question:
        next_question = last_host_question
        
    return " ".join(combined), next_question

def create_combined_dialogue_df(dataset_filepath, output_dir="output_results"):
    df = pd.read_csv(dataset_filepath, on_bad_lines='skip')
    df['QA_Sequence'] = df.apply(lambda row: combine_all_qa_pair(row), axis=1)
    df = df.drop(columns=['utt', 'speaker'])
    
    combined_file_path = os.path.join(output_dir, "QA_Sequence.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(combined_file_path, index=False)
    return df

def create_QA_Sequence_df_N_qa_pairs(dataset_filepath, N, output_dir="output_results"):
    df = pd.read_csv(dataset_filepath, on_bad_lines='skip')
    results = df.apply(lambda row: combine_N_qa_pairs_and_next_question(row, N), axis=1)
    df['QA_Sequence'], df['Actual_Question'] = zip(*results)
    df = df.drop(columns=['utt', 'speaker'])
    
    combined_file_path = os.path.join(output_dir, "QA_Sequence_and_next_question.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(combined_file_path, index=False)
    return df

def combine_csv_files(directory_path, output_file_name):
    dataframes = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)    
    combined_df.to_csv(output_file_name, index=False)
    print(f"Combined CSV saved to {output_file_name}")

# ------------- extract data section ------------- #

# given "ABC[XYZ]EFG", return "XYZ"
def extract_text_inside_brackets(text):
    match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if match:
        return match.group(1)
    return ""

# given "ABC{XYZ}EFG", return "XYZ"
def extract_text_inside_parentheses(text):
    match = re.search(r'\((.*?)\)', text)
    if match:
        return match.group(1)
    return "Error"

def stitch_csv_files(output_dir="output_results", final_output_file="all_results_concatenated.csv"):
    all_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.csv')]
    
    all_dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        all_dfs.append(df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_output_path = os.path.join(output_dir, final_output_file)
    final_df.to_csv(final_output_path, index=False)
    print(f"All CSV files stitched together into {final_output_path}")
    return final_df

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

def find_project_root(current_path, project_dir_name):
    while True:
        if project_dir_name in os.listdir(current_path):
            return os.path.join(current_path, project_dir_name)
        parent_dir = os.path.dirname(current_path)
        if current_path == parent_dir:
            raise ValueError(f"Project directory '{project_dir_name}' not found.")
        current_path = parent_dir

def calculate_gpt4_cost(prompt_file_path, response_file_path, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    PRICE_PER_1000_PROMPT_TOKENS = 0.00500
    PRICE_PER_1000_RESPONSE_TOKENS = 0.00500
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open(prompt_file_path, 'r') as prompt_file:
        prompts = prompt_file.readlines()

    with open(response_file_path, 'r') as response_file:
        responses = response_file.readlines()

    total_prompt_tokens = sum([len(tokenizer.encode(prompt)) for prompt in prompts])
    total_response_tokens = sum([len(tokenizer.encode(response)) for response in responses])

    prompt_cost = (total_prompt_tokens / 1000) * PRICE_PER_1000_PROMPT_TOKENS
    response_cost = (total_response_tokens / 1000) * PRICE_PER_1000_RESPONSE_TOKENS
    total_cost = prompt_cost + response_cost

    print(f"Total Prompt Tokens: {total_prompt_tokens}")
    print(f"Total Response Tokens: {total_response_tokens}")
    print(f"Total Prompt Cost: ${prompt_cost:.4f}")
    print(f"Total Response Cost: ${response_cost:.4f}")
    print(f"Total Cost: ${total_cost:.4f}")

    return total_prompt_tokens, total_response_tokens, prompt_cost, response_cost, total_cost

if __name__ == "__main__": 
    directory_path = 'output_results/gpt_batching/gpt4o_csv_outputs'
    output_file = 'output_results/gpt_batching/gpt4o_csv_outputs/gpt_all_interviews_combined.csv'
    combine_csv_files(directory_path, output_file)
