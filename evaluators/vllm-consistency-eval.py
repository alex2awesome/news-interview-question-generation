import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer
import torch
import re
from prompts import DIMENSION_OF_SIMILARITY_PROMPT
from helper_functions import extract_text_inside_brackets
# from vllm-type-classification import classify_question 
# ^include classify_question(model_name, messages) as a parameter later
from vllm import LLM, SamplingParams
import json

HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as config_file:
    config_data = json.load(config_file)
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['HF_HOME'] = HF_HOME

def load_vllm_model(model_name):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    model = LLM(
        model_name,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=HF_HOME,
        enforce_eager=True
    )
    return model

def vllm_infer(model_name, messages):
    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    output = model.generate(formatted_prompt, sampling_params)
    return output[0].outputs[0].text

def consistency_compare(model_name, messages):
    generated_text = vllm_infer(model_name, messages)
    print(f"generated_text: {generated_text}")

    similarity_scores_str = extract_text_inside_brackets(generated_text)
    similarity_scores_list = similarity_scores_str.split(', ')
    print(f'new sim score: {similarity_scores_list}')
    similarity_scores = [1 if score.lower() == 'yes' else 0 for score in similarity_scores_list]
    print(f'old sim score: {similarity_scores}')
    
    def similarity_score(scores): 
        return sum(scores)
    
    return similarity_score(similarity_scores)

if __name__ == "__main__":
    llm_question = "What are the main causes of climate change?"
    human_question = "Can you explain why the climate is changing?"
    transcript_context = "We are discussing environmental issues, particularly focusing on climate change and its causes."
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    prompt = DIMENSION_OF_SIMILARITY_PROMPT.format(
        transcript_context=transcript_context,
        LLM_question=llm_question,
        human_question=human_question
    )
    messages = [
        {"role": "system", "content": "You are a world-class annotator for question similarity."},
        {"role": "user", "content": prompt}
    ]
    
    sim_score = consistency_compare(model_name, messages)
    print(f'Total similarity score: {sim_score}')

    '''
    Future implementation: 
    1. Incorporate function call to classify_question(model_name, messages) 
    and include it as an input to make the LLM evalution function more robust.
    2. Set up a function that is capable of loading entire batch of data.
    '''