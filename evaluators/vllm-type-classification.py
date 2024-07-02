import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer
import torch
import re
from prompts import TAXONOMY, CLASSIFY_USING_TAXONOMY_PROMPT
from helper_functions import extract_text_inside_brackets
from vllm import LLM, SamplingParams
import json

HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
config_data = json.load(open('/project/jonmay_231/spangher/Projects/news-interview-question-generation/evaluators/config.json'))
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['HF_HOME'] = HF_HOME

def load_model_and_tokenizer(model_name):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    model = LLM(
        model_name,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=HF_HOME,
        enforce_eager=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_response(model, messages):
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    output = model.generate(formatted_prompt, sampling_params)
    return output[0].outputs[0].text

# Returns question type given relevant transcript context
def classify_question(transcript_section):
    prompt = CLASSIFY_USING_TAXONOMY_PROMPT.format(transcript_section=transcript_section)
    full_prompt = [
        {"role": "system", "content": "You are a world-class annotator for interview questions."},
        {"role": "user", "content": prompt}
    ]
    generated_text = generate_response(model, full_prompt).strip()
    print(generated_text)

    question_type = extract_text_inside_brackets(generated_text)

    if question_type in TAXONOMY:
        return question_type
    else:
        return "Unknown question type"

def process_transcripts(database):  # Needs to be transferred to helper_functions.py
    """
    Parameters:
        database (list): A list of interview transcript sections.

    Returns:
        list: A list of classified question types.
    """
    classified_questions = []
    for transcript in database:
        question_type = classify_question(transcript)
        classified_questions.append(question_type)
    return classified_questions

if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_name)
    transcript = """ 

    RACHEL MARTIN, HOST:

    Howard Lutnick is the CEO of the financial firm Cantor Fitzgerald. His company occupied the 101st to 105th floors of One World Trade Center. On September 11, 2001, he lost his brother and 658 of his colleagues. Lutnick survived and vowed to keep the firm alive. Now, 15 years later, he is still the CEO. And he joins us on the line from New York. Thank you so much for taking the time.

    HOWARD LUTNICK: Hey. It's my pleasure, Rachel.

    MARTIN: I'm sure there are a lot of moments and conversations that stand out from that first 24-hour period. But could I ask you to share one or two that stick with you?

    LUTNICK: Sure. So the night of September 11, I didn't really know who was alive and who wasn't alive. So we had a conference call. It was about 10 o'clock at night. And my employees called in. And I said, look, we have two choices.

    We can shut the firm down and go to our friends' funerals. Remember, that would be 20 funerals a day every day for 35 straight days. And I've got to tell you, If'm not really interested in going to work. All I want to do is climb under the covers and hug my family.

    But if we are going to go to work, we're going to do it to take care of our friends' families. So what do you want to do? You guys want to shut it down? Or do you want to work harder than you've ever worked before in your life? And that was the moment where the company survived.

    MARTIN: You weren't there on that morning.
    """
    q_type = classify_question(model, tokenizer, transcript)
    print(q_type)