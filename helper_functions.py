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
import ast
from tqdm.auto import tqdm
import torch.distributed as dist
from collections import defaultdict

## global variables
_tokenizer = None
_openai_model_name = 'gpt-4o'
_client = None
_model = defaultdict(lambda: None)

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
# def load_vllm_model(model_name="meta-llama/meta-llama-3.1-70b-instruct"):
#     global _model
#     if _model[model_name] is not None:
#         torch.cuda.empty_cache()
#         torch.cuda.memory_summary(device=None, abbreviated=False)

#         model = LLM(
#             model_name,
#             dtype=torch.float16,
#             tensor_parallel_size=torch.cuda.device_count(),
#             enforce_eager=True,
#             max_model_len=60_000
#         )

#         memory_allocated = torch.cuda.memory_allocated()
#         memory_reserved = torch.cuda.memory_reserved()
        
#         print(f"Model {model_name} loaded. Memory Allocated: {memory_allocated / (1024 ** 3):.2f} GB")
#         print(f"Model {model_name} loaded. Memory Reserved: {memory_reserved / (1024 ** 3):.2f} GB")
#         _model[model_name] = model
#     return _model[model_name]


def load_vllm_model(model_name="meta-llama/meta-llama-3.1-70b-instruct"):
    global _model
    if _model[model_name] is None:
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)

        model = LLM(
            model_name,
            dtype=torch.float16,
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=True,
            max_model_len=60_000
        )

        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        
        print(f"Model {model_name} loaded. Memory Allocated: {memory_allocated / (1024 ** 3):.2f} GB")
        print(f"Model {model_name} loaded. Memory Reserved: {memory_reserved / (1024 ** 3):.2f} GB")
        _model[model_name] = model
    else:
        model = _model[model_name]
    _ = initialize_tokenizer(model_name) # cache the tokenizer 
    return model


def load_model(model_name):
    """Generic function to either load a VLLM model or a client (e.g. OpenAI)."""
    if "gpt" in model_name:
        global _openai_model_name
        _openai_model_name = model_name
        return get_openai_client()
    else:
        return load_vllm_model(model_name)


def initialize_tokenizer(model_name=None):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


# for batching
def vllm_infer_batch(messages_batch, model):
    tokenizer = initialize_tokenizer()
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    outputs = model.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def query_openai(message, model=None):
    if model is None:
        model = get_openai_client()

    completion = model.chat.completions.create(
        model=_openai_model_name,
        messages=message
    )
    return completion.choices[0].message.content


def openai_infer_batch(messages_batch, model, verbose=False):
    return [query_openai(messages, model=model) for messages in tqdm(messages_batch, desc="OpenAI Inference", disable=not verbose)]


def infer_batch(messages_batch, model, verbose=False):
    if isinstance(model, OpenAI):
        return openai_infer_batch(messages_batch, model, verbose)
    else:
        return vllm_infer_batch(messages_batch, model)

import random
from scipy.stats import beta

# parameters of the beta distribution 
PERSONA_DICT = {
    'anxious': [(3, 7), (4, 6), (5, 5), (7, 3.5), (9, 2)],
    'avoidant': [(2.5, 6.5), (5, 6.75), (7, 7), (7.25, 4), (7.5, 1.5)],
    'adversarial': [(1, 9), (2, 8.5), (4, 8), (8, 7.5), (9, 7)],
    'defensive': [(4, 8), (6, 7), (8.5, 6.5), (8.5, 4.25), (8.5, 2)],
    'straightforward': [(2, 5.5), (4, 5.5), (5.5, 5.5), (7.75, 4), (10, 2.5)],
    'poor explainer': [(3.5, 7.5), (5.5, 7.5), (7.5, 7.5), (7.25, 4.35), (7, 1.2)],
    'dominating': [(2, 6), (4, 6), (6, 6), (7.5, 3.8), (8, 1.6)],
    'clueless': [(3.6, 8.0), (4.8, 6.5), (5, 5), (6.55, 3.3), (8.1, 1.6)]
}
UNIFORM_BETA_PARAMS = (1, 1)

def sample_proportion_from_beta(persona, persuasion_level, game_level="advanced"):
    """
    Sample a proportion from the beta distribution based on persona and persuasion level.

    Parameters:
        - persona (str): The persona of the source (e.g., 'anxious', 'dominant', etc.).
        - persuasion_level (int): The level of persuasion (1-5).

    Returns:
        float: A proportion between 0 and 1 sampled from the beta distribution.
    """
    if game_level == "basic":
        return 1 
    elif game_level == "intermediate":
        return beta.rvs(UNIFORM_BETA_PARAMS[0], UNIFORM_BETA_PARAMS[1])
    else:
        a, b = PERSONA_DICT[persona][persuasion_level - 1]
        proportion = beta.rvs(a, b)
        proportion = max(0.0, min(1.0, proportion))
        return proportion


def get_relevant_info_items(
        info_item_numbers, 
        info_items_dict, 
        persona, 
        persuasion_level, 
        used_info_items, 
        game_level="advanced"
):
    """
    Retrieve relevant information items based on the given parameters.

    This function identifies and returns a subset of information items that are relevant to the current 
    interview context. It considers the persona and persuasion level to determine the proportion of 
    information items to return. The function also ensures that previously used information items are not 
    selected again.

    Parameters:
        - info_item_numbers (list of int): A list of information item numbers that are relevant to a previously asked question.
        - info_items_dict (dict): A dictionary mapping all information item keys to their content.
        - persona (str): The persona of the source (e.g., 'anxious', 'dominant', etc.).
        - persuasion_level (int): The level of persuasion (0, 1, or 2).
        - used_info_items (set of int): A set of information item numbers that have already been used.

    Returns:
        tuple: A tuple containing:
            - str: A formatted string of the selected information items or a message if no items are available.
            - list of int: A list of numbers representing the selected information items.
    """
    # Initialize a list to store available information items
    available_items = []
    
    # Iterate over the provided information item numbers
    for num in info_item_numbers:
        # Check if the item has not been used
        if num not in used_info_items:
            # Construct the key for the information item
            key = f"Information Item #{num}"
            # Retrieve the content of the information item from the dictionary
            content = info_items_dict.get(key.lower(), '')
            # If content exists, add it to the available items list
            if content:
                available_items.append((num, f"{key}: {content}"))
    
    # Calculate the total number of items and the number of used items
    N = len(info_item_numbers)
    k = len([num for num in info_item_numbers if num in used_info_items])  # num of used items
    available = N - k  # Calculate the number of available items

    # Check if all relevant information has been used
    if available == 0:
        return "All relevant information has been given to the interviewer already!", []  
    
    # Check if there are no available items
    if not available_items:
        return "No information items align with the question", []
    
    # Sample a proportion of items to return based on persona and persuasion level
    proportion = sample_proportion_from_beta(persona, persuasion_level, game_level=game_level)
    num_items_to_retrieve = int(proportion * N)
    
    # Adjust the number of items to return if it exceeds available items
    num_items_to_retrieve = min(num_items_to_retrieve, available)

    # If there are items to return, sample and format them
    if num_items_to_retrieve > 0:
        sampled = random.sample(available_items, num_items_to_retrieve)
        sampled_items_str = "\n".join(item[1] for item in sampled)
        sampled_item_numbers = [item[0] for item in sampled]
        return sampled_items_str, sampled_item_numbers
    
    # If no items are selected, return a default message
    else:
        return "No relevant information for this question. Feel free to ramble and say nothing important.", []


def get_all_relevant_info_items(info_item_numbers, info_items_dict):
    """
    Retrieve all relevant information items based on provided item numbers.

    This function takes a list of information item numbers and a dictionary containing
    information items. It constructs a list of all relevant information items that
    correspond to the provided item numbers.

    Parameters:
    - info_item_numbers (list of int): A list of numbers representing the information items to retrieve.
    - info_items_dict (dict): A dictionary where keys are information item identifiers and values are the content.

    Returns:
    - str: A formatted string containing all relevant information items. If no items align with the question,
            a default message is returned.
    """
    # Initialize a list to store all relevant information items
    all_items = []
    
    # Iterate over the provided information item numbers
    for num in info_item_numbers:
        # Construct the key for the information item
        key = f"information item #{num}".lower()
        # Retrieve the content of the information item from the dictionary
        content = info_items_dict.get(key, '')
        # If content exists, add it to the all items list
        if content:
            all_items.append(f"Information Item #{num}: {content}")

    # Check if no relevant information items were found
    if not all_items:
        return "No information items align with the question"

    # Return a formatted string of all relevant information items
    return "\n".join(all_items) if all_items else "No information items align with the question"



def robust_load(x):
    try:
        return json.loads(x)
    except:
        try:
            return ast.literal_eval(x)
        except:
            raise ValueError(f"Could not load {x}")
    


# for single-use testing only
def vllm_infer(messages, model_name="meta-llama/meta-llama-3.1-70b-instruct"):
    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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

# setup openai API
if 'OPENAI_API_KEY' not in os.environ:
    OPENAI_KEY_PATH = os.path.expanduser('~/.openai-api-key.txt')
    if not os.path.exists(OPENAI_KEY_PATH):
        OPENAI_KEY_PATH = os.path.expanduser('~/.openai-isi-project-key.txt')
    if not os.path.exists(OPENAI_KEY_PATH):
        raise ValueError("OpenAI API key file not found.")
    os.environ['OPENAI_API_KEY'] = open(OPENAI_KEY_PATH).read().strip()

def get_openai_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


##
## 
## Game Simulation Helper Functions
## 
## 
def generate_INTERVIEWER_response_batch(prompts, model=None):
    messages_batch = [
        [
            {"role": "system", "content": "You are a journalistic interviewer."},
            {"role": "user", "content": prompt}
        ] for prompt in prompts
    ]
    return infer_batch(messages_batch, model)


def generate_SOURCE_response_batch(prompts, model=None):
    messages_batch = [
        [
            {"role": "system", "content": "You are a guest getting interviewed."},
            {"role": "user", "content": prompt}
        ] for prompt in prompts
    ]
    return infer_batch(messages_batch, model)


# from "Information Item {integer}", extracts {integer}
def extract_information_item_numbers(response):
    """
    Extracts all integers from a given response string that are associated with 
    the phrase "Information Item". The function looks for patterns in the form 
    of "Information Item {integer}" or "Information Item #{integer}" and returns 
    a list of the extracted integers.

    Args:
        response (str): The input string from which to extract information item numbers.

    Returns:
        list: A list of integers representing the extracted information item numbers.
    """
    return [int(num) for num in re.findall(r'(?i)information item #?(\d+)', response)]


# return total num of matches to "Information item #{integer}"
def count_information_items(info_items_text):
    """
    Counts the total number of occurrences of the pattern "Information item #{integer}" 
    in the provided text. The function is case-insensitive and looks for both 
    "Information item {integer}" and "Information item #{integer}" patterns.

    Args:
        info_items_text (str): The text in which to count the occurrences of 
        information item patterns.

    Returns:
        int: The total number of matches found for the pattern "Information item #{integer}".
    """
    return len(re.findall(r'(?i)information item #?\d+', info_items_text))



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

def remove_text_before_keyword(text, keyword="Source biography:"):
    """
    Removes all text before the specified keyword in the input string.

    Parameters:
    - text (str): The original text.
    - keyword (str): The keyword to search for.

    Returns:
    - str: The cleaned text starting from the keyword.
    """
    if pd.isna(text):
        return text
    index = text.find(keyword)
    if index != -1:
        return text[index:]
    else:
        return text

# ------------- extract data section ------------- #

# given "ABC[XYZ]EFG", return "XYZ"
def extract_text_inside_brackets(text):
    match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if match:
        return match.group(1)
    return ""


def parse_python_dict(text):
    """
    Parses a Python dictionary from a string.

    Parameters:
    - text (str): The string containing the dictionary.

    Returns:
    - dict: The parsed dictionary if found, otherwise None.
    """
    match = re.search(r'\{.*:.*\}', text, re.DOTALL)
    if match:
        dict_str = match.group(0)
        try:
            return ast.literal_eval(dict_str)
        except (SyntaxError, NameError):
            return {}
    else:
        print(f"Error: No dictionary found in text: {text}")
        return {}   


# given "ABC{XYZ}EFG", return "XYZ"
def extract_text_inside_parentheses(text):
    match = re.search(r'\((.*?)\)', text)
    if match:
        return match.group(1)
    return "Error"


import glob
def stitch_csv_files(output_dir="output_results", final_output_file="all_results_concatenated.csv"):
    json_files = glob.glob(os.path.join(output_dir, '*.jsonl'))
    json_dfs = list(map(lambda x: pd.read_json(x, lines=True), json_files))
    csv_files = glob.glob(os.path.join(output_dir, '*.csv'))
    csv_dfs = list(map(lambda x: pd.read_csv(x), csv_files))
    all_dfs = json_dfs + csv_dfs
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_output_path = os.path.join(output_dir, final_output_file)
    if final_output_file.endswith('.csv'):
        final_df.to_csv(final_output_path, index=False)
    elif final_output_file.endswith('.jsonl'):
        final_df.to_json(final_output_path, orient='records', lines=True)
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

def log_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        print(f"GPU {i}: Allocated {allocated:.2f} GB, Reserved {reserved:.2f} GB")



if __name__ == "__main__": 
    directory_path = 'output_results/gpt_batching/gpt4o_csv_outputs'
    output_file = 'output_results/gpt_batching/gpt4o_csv_outputs/gpt_all_interviews_combined.csv'
    combine_csv_files(directory_path, output_file)
