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

def get_outline_parts(id, split_outlines):
    outline_row = split_outlines[split_outlines['id'] == id]
    if not outline_row.empty:
        return {
            'outline_statement': outline_row['outline_statement'].values[0],
            'interview_goals': outline_row['interview_goals'].values[0],
            'general_questions': outline_row['general_questions'].values[0]
        }
    return {
        'outline_statement': 'N/A',
        'interview_goals': 'N/A',
        'general_questions': 'N/A'
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--data_dir', type=str, default='../dataset')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()

    task_dataset_file_path = os.path.join(args.data_dir, 'task_dataset.csv')
    task_dataset = pd.read_csv(task_dataset_file_path)

    outline_file_path = os.path.join(args.data_dir, 'transcripts_with_split_outlines_fixed.csv')
    split_outlines = pd.read_csv(outline_file_path)
    # if args.start_idx is not None and args.end_idx is not None:
    #     task_dataset = task_dataset.iloc[0:1]
    task_dataset = task_dataset.iloc[0:1000]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    message_batches = []
    #qu_pairs = task_dataset['Input (first k qa_pairs)'].tolist()

    id_and_pairs = zip(task_dataset['id'].tolist(), task_dataset['Input (first k qa_pairs)'].tolist())
    
    messages = []
    for (transcript_id, qu_pair) in id_and_pairs:
        outline_parts = get_outline_parts(transcript_id, split_outlines)

        prompt = f"""
            You are a journalist's assistant and your task is to predict the next question that will follow in an interview. I will give you a piece of the interview transcript as well as the motivation behind the interview.
            
            Think about this step-by-step:
            How did the previous response of the interview address the question?
            Did it answer the question or do we need to ask a clarifying question?
            What other components does this story need?/what more information does this source have?
            Do we need ask a follow up?
            
            I want you to predict the next question the journalist will ask. Separately, I want you to give me the reason/purpose for asking this question.
            Make sure that you are recognizing the interviewee's last comment rather than immediately moving on and asking a question. This should sound personal and should sound like you care about what the interviewee has to say. This can simply acknowledge what they said. 
            Be mindful of the transitions so it sounds as personal and structured as possible.
            
            Respond the purpose of the question in parenthes. For example (The purpose of this question...).
            Respond with the suggested question in brackets. For example [What is...].

            ```{outline_parts['outline_statement']}```

            Here is an outline of your goals and top questions you want to ask for the interview: 
            ```{outline_parts['interview_goals']}```
            ```{outline_parts['general_questions']}```
            Here is a few back and forth QAs that will lead you into your output of responding and asking the next question: 
            ```{qu_pair}```

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
        args.end_idx = len(task_dataset)

    start_idx = args.start_idx
    end_idx = start_idx + BATCH_SIZE

    final_responses = []

    def extract_text_inside_brackets(text):
        match = re.search(r'\[(.*?)\]', text)
        if match:
            return match.group(1)
        return "Error"

    def extract_text_inside_parentheses(text):
        match = re.search(r'\((.*?)\)', text)
        if match:
            return match.group(1)
        return "Error"

    outline_only_question_gen = task_dataset.copy()
    outline_only_question_gen['suggested_question'] = None
    outline_only_question_gen['purpose'] = None 

    for batch_idx, messages in enumerate(tqdm(message_batches)):
        outputs = model.generate(messages, sampling_params)
        
        start_row = batch_idx * BATCH_SIZE
        end_row = start_row + len(outputs)
        for i, output in enumerate(outputs):
            response = unicodedata.normalize('NFKC', output.outputs[0].text)
            question = extract_text_inside_brackets(response)
            purpose = extract_text_inside_parentheses(response)

            row_index = start_row + i
            if row_index < len(outline_only_question_gen):
                outline_only_question_gen.at[row_index, 'suggested_question'] = question
                outline_only_question_gen.at[row_index, 'purpose'] = purpose

    outline_only_question_gen.to_csv(os.path.join(args.data_dir, 'outline_only_question_gen_1000.csv'), index=False)
    print("Completed.")