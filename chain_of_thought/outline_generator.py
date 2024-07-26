# import pandas as pd
# import numpy as np
# from tqdm.auto import tqdm
# import unicodedata
# import os
# import json
# import torch
# import logging
# import re

# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )

# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer

# HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
# config_data = json.load(open('/project/jonmay_231/spangher/Projects/news-interview-question-generation/configs/config.json'))
# os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
# os.environ['HF_HOME'] = HF_HOME

# BATCH_SIZE = 100

# def load_model(model: str):
#     torch.cuda.empty_cache()
#     torch.cuda.memory_summary(device=None, abbreviated=False)
#     model = LLM(
#         model,
#         dtype=torch.float16,
#         tensor_parallel_size=torch.cuda.device_count(),
#         download_dir=HF_HOME,
#         enforce_eager=True
#     )
#     return model

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
#     parser.add_argument('--data_dir', type=str, default='../dataset')
#     parser.add_argument('--start_idx', type=int, default=None)
#     parser.add_argument('--end_idx', type=int, default=None)
#     args = parser.parse_args()

#     csv_file_path = os.path.join(args.data_dir, 'final_dataset.csv')

#     df = pd.read_csv(csv_file_path)
#     # if args.start_idx is not None and args.end_idx is not None:
#     #     df = df.iloc[0:1]

#     df = df.iloc[0:1]

#     print("Column names in DataFrame:", df.columns)

#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

#     message_batches = []
#     dialogues = df['combined_dialogue'].tolist()
    
#     messages = []
#     for dialogue in dialogues:
#         prompt = f"""
#             Given the following interview transcript, can you provide a brief outline that a journalist might have used before conducting the interview? The outline should include:

#             Interview goals (in paragraph form).
#             Example interview goals:
#             **Interview Goals:**
#             The goal of the interview is to explore FCC Chairman Ajit Pai's perspective on net neutrality and his proposed changes to the current regulations. The interview seeks to understand his rationale for wanting to shift from Title II regulations to a more market-driven approach, address concerns from critics about potential negative impacts on consumer rights and small providers, and discuss his vision for the future of internet regulation.

#             An outline statement (in paragraph form). Estimated length (in minutes).
#             Example outline statement:
#             **Outline Statement:**
#             "You’re about to interview Ajit Pai, the Chairman of the Federal Communications Commission. The goals of the interview are to delve into his views on net neutrality, the reasons behind his proposed regulatory changes, and the potential implications for consumers and internet service providers. This will be a 10-question interview that will last approximately 15 minutes."
            
#             A few general questions or topics in bullet points (no more than 6).
#             Example questions:
#             **General Questions/Talking Points:**
#             Can you explain why you believe the current Title II regulations are not suitable for today's internet marketplace?
#             What specific benefits do you foresee in reverting to the Clinton-era approach to internet regulation?
#             Critics argue that loosening regulations could harm consumers by allowing ISPs to prioritize certain content. How do you respond to these concerns?
#             How do you ensure that smaller internet service providers are not disproportionately affected by reduced regulations?
#             In what ways can consumers hold ISPs accountable if net neutrality protections are weakened?
#             What is your stance on treating the internet as a utility, given significant public support for this approach?

#             Do not include anything except for the interview goals, an outline statement, and a few general questions.
#             The entirety of the outline, including the questions, should be placed in only one set of brackets. **Do not use any other brackets.**
            
#             Here's the transcript: 
#             ```{dialogue}```
#         """
#         message = [
#             {
#                 "role": "system",
#                 "content": "You are an experienced journalist.",
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             },
#         ]
#         formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

#         messages.append(formatted_prompt)

#         if len(messages) >= BATCH_SIZE:
#             message_batches.append(messages)
#             messages = []

#     if messages:
#         message_batches.append(messages)

#     model = load_model(args.model)
#     sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
#     if args.start_idx is None:
#         args.start_idx = 0
#     if args.end_idx is None:
#         args.end_idx = len(df)

#     start_idx = args.start_idx
#     end_idx = start_idx + BATCH_SIZE

#     final_responses = []

#     def extract_text_inside_brackets(text):
#         match = re.search(r'\[(.*?)\]', text)
#         if match:
#             return match.group(1)
#         return "Error"

#     for messages in tqdm(message_batches):
#         fname = f'responses.txt'
#         outputs = model.generate(messages, sampling_params)
#         with open(fname, 'wb') as file:
#             for output in outputs:
#                 response = output.outputs[0].text
#                 response = unicodedata.normalize('NFKC', response)
#                 if response:
#                     file.write(response.encode('utf-8'))
#                     file.write(b'\n\n')
#                     extracted_response = extract_text_inside_brackets(response)
#                     final_responses.append(response)

#     print(final_responses[0])

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--data_dir', type=str, default='../dataset')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()

    csv_file_path = os.path.join(args.data_dir, 'final_dataset.csv')

    df = pd.read_csv(csv_file_path)
    if args.start_idx is not None and args.end_idx is not None:
        df = df.iloc[0:1]

    #df = df.iloc[0:25]

    print("Column names in DataFrame:", df.columns)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    message_batches = []
    dialogues = df['combined_dialogue'].tolist()
    
    messages = []
    for dialogue in dialogues:
        prompt = f"""
            Given the following interview transcript, can you provide a brief outline that a journalist might have used before conducting the interview? The outline should include:

            Interview goals (in paragraph form).
            Example interview goals:
            **Interview Goals:**
            The goal of the interview is to explore FCC Chairman Ajit Pai's perspective on net neutrality and his proposed changes to the current regulations. The interview seeks to understand his rationale for wanting to shift from Title II regulations to a more market-driven approach, address concerns from critics about potential negative impacts on consumer rights and small providers, and discuss his vision for the future of internet regulation.

            An outline statement (in paragraph form). Estimated length (in minutes).
            Example outline statement:
            **Outline Statement:**
            "You’re about to interview Ajit Pai, the Chairman of the Federal Communications Commission. The goals of the interview are to delve into his views on net neutrality, the reasons behind his proposed regulatory changes, and the potential implications for consumers and internet service providers. This will be a 10-question interview that will last approximately 15 minutes."
            
            A few general questions or topics in bullet points (no more than 6).
            Example questions:
            **General Questions/Talking Points:**
            Can you explain why you believe the current Title II regulations are not suitable for today's internet marketplace?
            What specific benefits do you foresee in reverting to the Clinton-era approach to internet regulation?
            Critics argue that loosening regulations could harm consumers by allowing ISPs to prioritize certain content. How do you respond to these concerns?
            How do you ensure that smaller internet service providers are not disproportionately affected by reduced regulations?
            In what ways can consumers hold ISPs accountable if net neutrality protections are weakened?
            What is your stance on treating the internet as a utility, given significant public support for this approach?

            Do not include anything except for the interview goals, an outline statement, and a few general questions.
            The entirety of the outline, including the questions, should be placed in only one set of brackets. **Do not use any other brackets.**
            
            Here's the transcript: 
            ```{dialogue}```
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

    transcripts_with_outlines = df.copy()
    transcripts_with_outlines['outline'] = None 

    for batch_idx, messages in enumerate(tqdm(message_batches)):
        outputs = model.generate(messages, sampling_params)
        
        start_row = batch_idx * BATCH_SIZE
        end_row = start_row + len(outputs)
        transcripts_with_outlines.loc[start_row:end_row - 1, 'outline'] = [
            unicodedata.normalize('NFKC', output.outputs[0].text) for output in outputs
        ]

    transcripts_with_outlines.to_csv(os.path.join(args.data_dir, 'transcripts_with_outlines.csv'), index=False)
    print("Completed.")