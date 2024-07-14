import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import json
import torch
import logging
import unicodedata
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Set environment variables
def setup_hf_env():
    HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
    config_data = json.load(open('/project/jonmay_231/spangher/Projects/news-interview-question-generation/configs/config.json'))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
    os.environ['HF_HOME'] = HF_HOME
    return HF_HOME

#BATCH_SIZE = 100

# def load_model(model: str):
#     torch.cuda.memory_summary(device=None, abbreviated=False)
#     model = LLM(
#         model,
#         dtype=torch.float16,
#         tensor_parallel_size=torch.cuda.device_count(),
#         download_dir=HF_HOME,
#         enforce_eager=True,
#         gpu_memory_utilization=0.9
#     )
#     return model

# def process_model(model, message_batches):
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", chat=True)    
#     sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
#     responses = []
#     for messages in tqdm(message_batches):
#         formatted_prompts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
#         outputs = model.generate(formatted_prompts, sampling_params)
#         batch_responses = [output.outputs[0].text.lower().strip() for output in outputs]  # Assuming responses are lowercase and stripped
#         responses.extend(batch_responses)
#     return responses

def load_vllm_model(model_name="meta-llama/Meta-Llama-3-8B"):
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

def vllm_infer(messages, model_name="meta-llama/Meta-Llama-3-8B"):
    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    output = model.generate(formatted_prompt, sampling_params)
    return output[0].outputs[0].text

def make_prompts(source_df):
    messages = []
    for _, row in source_df.iterrows():
        combined_dialogue = row['combined_dialogue'].strip()
        prompt = f"""
        Analyze this interview transcript that is in the form of a dialogue:

        ```{combined_dialogue}``` #5-10 turns

        By reading through the dialogue, identify if this transcript is an informational interview between 2 people.
        An informational interview is an interview in which someone gains information from someone with experience in the field.
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
        messages.append(message)
    return messages

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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--data_file', type=str, default='dataset/combined_data.csv')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=1)
    args = parser.parse_args()

    # Load the original data
    original_df = pd.read_csv(args.data_file)

    # Slice the DataFrame to get only the specified rows
    test_df = original_df.iloc[args.start_idx:args.end_idx]

    # Create combined dialogue dataframe
    source_df = create_combined_dialogue_df(test_df)

    # Prepare messages
    messages = make_prompts(source_df)
    # message_batches = [messages[i:i + BATCH_SIZE] for i in range(0, len(messages), BATCH_SIZE)]

    # Load the model
    model = load_vllm_model(args.model)

    # Process model and get responses
    print(vllm_infer(messages, model_name="meta-llama/Meta-Llama-3-8B"))

if __name__ == "__main__":
    main()

