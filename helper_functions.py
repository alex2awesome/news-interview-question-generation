import tiktoken
import re
from openai import OpenAI
import os

def get_openai_client(key_file_path='~/.openai-api-key.txt'):
    key_path = os.path.expanduser(key_file_path)
    client = OpenAI(api_key=open(key_path).read().strip())
    return client

#given "ABC[XYZ]EFG", extracts XYZ
def extract_text_inside_brackets(text):
    match = re.search(r'\[(.*?)\]', text)
    if match:
        return match.group(1)
    else:
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