from transformers import AutoTokenizer
import torch
import re
from prompts import DIMENSION_OF_SIMILARITY_PROMPT
from helper_functions import extract_text_inside_brackets
from vllm import LLM, SamplingParams
import json
import os

HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
config_data = json.load(open('config.json'))
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['HF_HOME'] = HF_HOME

def load_model(model: str):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    model = LLM(
        model,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=HF_HOME,
        enforce_eager=True
    )
    return model

model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
model = load_model(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(model, messages):
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    output = model.generate(formatted_prompt, sampling_params)
    return output[0].outputs[0].text

def consistency_compare(LLM_question, human_question, transcript_context):
    prompt = DIMENSION_OF_SIMILARITY_PROMPT.format(
        transcript_context=transcript_context,
        LLM_question=LLM_question,
        human_question=human_question
    )
    full_prompt = [
        {"role": "system", "content": "You are a world-class annotator for question similarity."},
        {"role": "user", "content": prompt}
    ]
    generated_text = generate_response(model, full_prompt).strip()
    print(f'{generated_text}\n\n')

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
    sim_score = consistency_compare(llm_question, human_question, transcript_context)
    print(f'Total similarity score: {sim_score}')