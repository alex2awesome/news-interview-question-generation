from datasets import load_from_disk
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json
import torch
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

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

# def vllm_infer(messages, model_name="meta-llama/Meta-Llama-3-8B"):
#     model = load_vllm_model(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
#     output = model.generate(formatted_prompt, sampling_params)
#     return output.outputs[0].text

def vllm_infer(messages, model_name="meta-llama/Meta-Llama-3-8B"):
    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add logging or print statements to inspect messages
    logging.info(f"Messages received: {messages}")
    
    # Ensure messages is not empty and is in the expected format
    if not messages:
        raise ValueError("Messages list is empty or None.")
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Add logging to check formatted_prompt
    logging.info(f"Formatted prompt: {formatted_prompt}")
    
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    output = model.generate(formatted_prompt, sampling_params)
    return output.outputs[0].text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument('--data_dir', type=str, default='../dataset')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=1)
    args = parser.parse_args()

    # Relative path to the CSV file
    csv_file_path = os.path.join(args.data_dir, 'combined_data.csv')

    # Check if the file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File not found: {csv_file_path}")

    logging.info(f"Reading CSV file from {csv_file_path}")
    
    source_df = pd.read_csv(
        csv_file_path, nrows=args.end_idx
    ).iloc[args.start_idx:args.end_idx]

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
    def create_combined_dialogue_df(df, output_dir="../output_results"):
        df['combined_dialogue'] = df.apply(lambda row: combine_all_qa_pair(row), axis=1)
        combined_file_path = os.path.join(output_dir, "combined_data_with_dialogue.csv")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(combined_file_path, index=False)
        return df
    
    new_df = create_combined_dialogue_df(source_df)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    messages = []

    for _, row in new_df.iterrows():
        combined_dialogue = row['combined_dialogue'].strip()
        prompt = f"""
            Analyze this interview transcript that is in the form of a dialogue:

            ```{combined_dialogue}```

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
        
        # Log combined_dialogue to verify its content
        logging.info(f"Combined dialogue: {combined_dialogue}")
        
        messages.append(message)
    
    # Ensure messages is not empty
    if not messages:
        raise ValueError("Messages list is empty after iteration.")

    response = vllm_infer(messages)
    print(response)

