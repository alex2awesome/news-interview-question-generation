import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata
import os
import json
import torch
import logging
import re

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--data_dir', type=str, default='../dataset')
    parser.add_argument('--data_path', type=str, default='datasets/combined_data.csv')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()

    csv_file_path = os.path.join(args.data_dir, 'combined_data.csv')

    df = pd.read_csv(csv_file_path)
    if args.start_idx is not None and args.end_idx is not None:
        df = df.iloc[0:1]

    #df = df.iloc[0:10]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # Function to combine speaker and dialogue
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

    # Function to create combined dialogue dataframe
    def create_combined_dialogue_df(df, output_dir="output_results"):
        df['combined_dialogue'] = df.apply(lambda row: combine_all_qa_pair(row), axis=1)
        combined_file_path = os.path.join(output_dir, "combined_data_with_dialogue.csv")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(combined_file_path, index=False)
        return df

    combined_dialogue = create_combined_dialogue_df(df, output_dir="data_processing")

    message_batches = []
    dialogues = combined_dialogue['combined_dialogue'].tolist()
    
    messages = []
    for dialogue in dialogues:
        prompt = f"""
        Analyze this interview transcript that is in the form of a dialogue:

        ```{dialogue}```

        By reading through the dialogue, identify if this transcript is an informational interview between 2 people.
        Look for questions and make sure this is an interview, not a Q&A game.
        The interviewer should be asking questions, not engaging in a back-and-forth conversation with the interviewee.

        After analyzing, your final answer of just 'YES' or 'NO' should be in brackets.
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

    for messages in tqdm(message_batches):
        fname = f'responses.txt'
        outputs = model.generate(messages, sampling_params)
        with open(fname, 'wb') as file:
            for output in outputs:
                response = output.outputs[0].text
                response = unicodedata.normalize('NFKC', response)
                if response:
                    file.write(response.encode('utf-8'))
                    file.write(b'\n\n')
                    extracted_response = extract_text_inside_brackets(response)
                    final_responses.append(extracted_response)

    filtered_data = []

    for idx, response in enumerate(final_responses):
        if response == 'YES':
            filtered_data.append(df.iloc[idx])

    filtered_data = pd.DataFrame(filtered_data)

    os.makedirs("output_results", exist_ok=True)
    output_csv_path = os.path.join("output_results", "final_dataset.csv")
    filtered_data.to_csv(output_csv_path, index=False)

    print(f"Filtered Data saved successfully as {output_csv_path}")
    print(f"Length of the new DataFrame: {len(filtered_data)}")