# helper_functions.py

import tiktoken
import re
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import os
import torch

#hugging face environment setup for VLLM functionality
def setup_hf_env():
    HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
    config_data = json.load(open('/project/jonmay_231/spangher/Projects/news-interview-question-generation/configs/config.json'))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
    os.environ['HF_HOME'] = HF_HOME
    return HF_HOME

#vllm framework model loader
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

def batch_process(dataset, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    results = []
    '''
    function does:
    1. loads dataset
    2. iterates through dataset
    for data in dataset:
        messages = do something to data
        vllm_infer(model_name, messages)
    '''
    return results

#setup openai API
def get_openai_client(key_file_path='~/.openai-api-key.txt'):
    key_path = os.path.expanduser(key_file_path)
    client = OpenAI(api_key=open(key_path).read().strip())
    return client

#given "ABC[XYZ]EFG", extracts XYZ
def extract_text_inside_brackets(text):
    match = re.search(r'\[(.*?)\]', text)
    if match:
        return match.group(1)
    return 0

#given "ABC{XYZ}EFG", extracts XYZ
def extract_text_inside_parentheses(text):
    match = re.search(r'\((.*?)\)', text)
    if match:
        return match.group(1)
    return 0

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