import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from vllm import LLM, SamplingParams
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import load_vllm_model, initialize_tokenizer
from game_sim_prompts import get_info_items_prompt

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

def extract_information_items(transcripts, model, tokenizer):
    information_items = []

    for transcript in transcripts:
        prompt = get_info_items_prompt(transcript)
        response = generate_vllm_response(prompt, "You are an AI extracting key information items from an interview transcript", model, tokenizer)
        information_items.append(response)

    return information_items

# ---- batch use ---- #
def vllm_infer_batch(messages_batch, model, tokenizer):
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    outputs = model.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

def generate_vllm_info_items_batch(prompts, model, tokenizer):
    messages_batch = [
            [
            {"role": "system", "content": "You are an AI extracting key information items from an interview transcript"}, 
            {"role": "user", "content": prompt}
            ] 
            for prompt in prompts
        ]
    return vllm_infer_batch(messages_batch, model, tokenizer)

def extract_information_items_batch(transcripts, model, tokenizer, batch_size=100):
    information_items = []
    
    for i in range(0, len(transcripts), batch_size):
        batch = transcripts[i:i+batch_size]
        prompts = [get_info_items_prompt(transcript) for transcript in batch]
        batch_responses = generate_vllm_info_items_batch(prompts, model, tokenizer)
        information_items.extend(batch_responses)
    
    return information_items

# ----------- #
def process_info_items(df, model_name="meta-llama/Meta-Llama-3-70B-Instruct", output_dir="output_results/game_sim/info_items"):
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    df['info_items'] = extract_information_items_batch(df['combined_dialogue'], model, tokenizer)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "final_df_with_info_items.csv")
    df.to_csv(output_file_path, index=False)
    print(f"csv file saved to {output_file_path}")
    return df

if __name__ == "__main__": 
    # final_dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/dataset/final_dataset.csv"
    # df = pd.read_csv(final_dataset_path)
    # df = df.head(100)
    # print(df)

    # df_with_info_items = process_info_items(df, model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    # print(df_with_info_items)

    test_file = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/game_sim/info_items/final_df_with_info_items.csv"
    test_df = pd.read_csv(test_file)
    
    print(f"interview: {test_df['combined_dialogue'].iloc[2]}")
    print(f"info items: {test_df['info_items'].iloc[2]}")
    print(test_df["id"].iloc[2])

    # checked that it works with the 8B model? y/n: y 
    # (validated by michael)
