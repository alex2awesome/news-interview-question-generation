import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata
import os
import json
import torch
import logging
import re
import ast

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
config_data = json.load(open('/project/jonmay_231/spangher/Projects/news-interview-question-generation/configs/config.json'))
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['HF_HOME'] = HF_HOME

BATCH_SIZE = 100
WRITE_BATCH_SIZE = 100  # Number of responses to write to the file at once

def load_model(model: str):
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    model = LLM(
        model,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=HF_HOME,
        enforce_eager=True
    )
    return model

def ensure_valid_json(response):
    response = response.replace('\n', '').replace('\r', '')

    response = response.strip()

    if response.endswith(']'):
        if not response.endswith(']}'):
            response = response[:-1] + ']}'
    elif not response.endswith(']}'):
        response += ']}'

    if not response.startswith('{'):
        response = '{' + response

    return response

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--data_dir', type=str, default='../dataset')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()

    csv_file_path = os.path.join(args.data_dir, 'transcripts_with_outlines.csv')

    df = pd.read_csv(csv_file_path)
    if args.start_idx is not None and args.end_idx is not None:
        df = df.iloc[0:1]

    #df = df.iloc[0:1000]

    #print("Column names in DataFrame:", df.columns)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    message_batches = []
    outlines = df['outline'].tolist()
    
    messages = []
    for outline in outlines:
        prompt = f"""
            Given the following interview outline, please return a JSON object containing exactly three fields in the specified format:

            1. **Interview Goals:** This should be a clear, concise statement about the goals of the interview.
            2. **Outline Statement:** This should summarize the key points or topics of the interview.
            3. **General Questions:** This should be a list of questions to ask during the interview.

            Ensure that your response is in the following format:

            {{
                "Interview Goals": "The goal of this interview is to explore...",
                "Outline Statement": "You're about to interview...",
                "General Questions": [
                    "What are the current statistics on HIV/AIDS?",
                    "What are the challenges faced?",
                    "What role do you see for government?"
                ]
            }}


            Note:
            - The keys should be "Interview Goals", "Outline Statement", and "General Questions".
            - Ensure that the values are strings (for the first two keys) and a list of strings (for the third key).
            - Make sure to use proper JSON syntax with double quotes and avoid any additional text or formatting.
            - Do not respond with anything except for the Python list. For example, do not start the response with "Here is the list in the specified format:".
            - The output must be valid JSON and parsable using `json.loads`.
            - If there are DOUBLE quotes within the strings of the interview statement or the outline statement, or general questions, ensure that the double quotes are properly escaped (\").
            - DO NOT escape SINGLE quotes (').

            Here's the outline:
            ```{outline}```
        """
        message = [
            {
                "role": "system",
                "content": "You are an experienced journalist.",
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
        formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        messages.append(formatted_prompt)

        if len(messages) >= BATCH_SIZE:
            message_batches.append(messages)
            messages = []

    if messages:
        message_batches.append(messages)

    model = load_model(args.model)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(df)

    start_idx = args.start_idx
    end_idx = start_idx + BATCH_SIZE

    final_responses = []

    def extract_text_inside_brackets(text):
        match = re.search(r'\[(.*?)\]', text)
        if match:
            return match.group(1)
        return "Error"

    transcripts_with_split_outlines = df.copy()
    transcripts_with_split_outlines['interview_goals'] = None
    transcripts_with_split_outlines['outline_statement'] = None
    transcripts_with_split_outlines['general_questions'] = None

    for batch_idx, messages in enumerate(tqdm(message_batches)):
        outputs = model.generate(messages, sampling_params)
        
        start_row = batch_idx * BATCH_SIZE
        end_row = start_row + len(outputs)
        for i, output in enumerate(outputs):
            response = unicodedata.normalize('NFKC', output.outputs[0].text)
            response = ensure_valid_json(response)
            try:
                response_dict = json.loads(response)
                transcripts_with_split_outlines.at[start_row + i, 'interview_goals'] = response_dict.get("Interview Goals", "Error")
                transcripts_with_split_outlines.at[start_row + i, 'outline_statement'] = response_dict.get("Outline Statement", "Error")
                transcripts_with_split_outlines.at[start_row + i, 'general_questions'] = response_dict.get("General Questions", "Error")
                #print(f"success: {response}")
            except json.JSONDecodeError:
                print(f"Failed to decode JSON: {response}")
                transcripts_with_split_outlines.at[start_row + i, 'interview_goals'] = "Error"
                transcripts_with_split_outlines.at[start_row + i, 'outline_statement'] = "Error"
                transcripts_with_split_outlines.at[start_row + i, 'general_questions'] = "Error"

    transcripts_with_split_outlines.drop(['program', 'date', 'url', 'title', 'summary'], axis=1, inplace=True)

    transcripts_with_split_outlines.to_csv(os.path.join(args.data_dir, 'transcripts_with_split_outlines.csv'), index=False)
    print("Completed.")