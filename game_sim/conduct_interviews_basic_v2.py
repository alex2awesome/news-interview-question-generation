# conduct_interviews_basic_v2.py
import os
import sys
import re
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import load_vllm_model, initialize_tokenizer, extract_text_inside_brackets, stitch_csv_files, find_project_root
from game_sim.game_sim_prompts import get_source_prompt_basic, get_source_starting_prompt, get_source_ending_prompt, get_source_specific_info_items_prompt, get_interviewer_prompt, get_interviewer_starting_prompt, get_interviewer_ending_prompt
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

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

def conduct_interview(initial_prompt, num_turns, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)
    conversation_history = initial_prompt
    print("Initial Prompt:")
    print(conversation_history)
    for turn in range(num_turns):
        interviewer_prompt = get_interviewer_prompt(conversation_history, "", "straightforward")
        interviewer_question = generate_vllm_response(interviewer_prompt, "You are a journalistic interviewer.", model, tokenizer)
        conversation_history += f"\nInterviewer: {interviewer_question}"
        print("\nInterviewer: " + interviewer_question)
        source_prompt = get_source_prompt_basic(conversation_history, "", "honest")
        interviewee_response = generate_vllm_response(source_prompt, "You are a guest getting interviewed.", model, tokenizer)
        conversation_history += f"\nInterviewee: {interviewee_response}"
        print("\nInterviewee: " + interviewee_response)
    print("\nFinal Conversation:\n" + conversation_history)
    return conversation_history

# ---- batch use ---- #
def vllm_infer_batch(messages_batch, model, tokenizer):
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    outputs = model.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

def generate_vllm_INTERVIEWER_response_batch(prompts, model, tokenizer):
    messages_batch = [
        [
            {"role": "system", "content": "You are a journalistic interviewer."},
            {"role": "user", "content": prompt}
        ] for prompt in prompts
    ]
    return vllm_infer_batch(messages_batch, model, tokenizer)

def generate_vllm_SOURCE_response_batch(prompts, model, tokenizer):
    messages_batch = [
        [
            {"role": "system", "content": "You are a guest getting interviewed."},
            {"role": "user", "content": prompt}
        ] for prompt in prompts
    ]
    return vllm_infer_batch(messages_batch, model, tokenizer)

# regex to match "Information Item {integer}" and extract the integer
def extract_information_item_numbers(response):
    return [int(num) for num in re.findall(r'(?i)information item #?(\d+)', response)]

# Regular expression to match "Information item #{integer}"
def count_information_items(info_items_text):
    return len(re.findall(r'(?i)information item #?\d+', info_items_text))

def log_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        print(f"GPU {i}: Allocated {allocated:.2f} GB, Reserved {reserved:.2f} GB")

def conduct_basic_interviews_batch(num_turns, df, interviewer_strategy="straightforward", interviewer_model_name="meta-llama/Meta-Llama-3-70B-Instruct", source_model_name="meta-llama/Meta-Llama-3-70B-Instruct", batch_size=20, output_dir="output_results/game_sim/conducted_interviews_basic"):
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = len(df)
    unique_info_item_counts = [0] * num_samples
    total_info_item_counts = [count_information_items(info_item) for info_item in df['info_items']]

    current_conversations = [""] * num_samples
    unique_info_items_sets = [set() for _ in range(num_samples)]

    for turn in range(num_turns):
        #### Interviewer
        print(f"Turn {turn + 1}: Generating interviewer questions")
        interviewer_model = load_vllm_model(interviewer_model_name)
        interviewer_tokenizer = initialize_tokenizer(interviewer_model_name)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_df = df.iloc[start_idx:end_idx].copy()
            outlines = batch_df['outlines']

            if turn == 0: 
                interviewer_prompts = [
                    get_interviewer_starting_prompt(outline, num_turns, interviewer_strategy)
                    for outline in outlines
                ]
            elif turn == num_turns - 1:
                interviewer_prompts = [
                    get_interviewer_ending_prompt(current_conversation, outline, interviewer_strategy)
                    for current_conversation, outline in zip(current_conversations, outlines)
                ]
            else:
                interviewer_prompts = [
                    get_interviewer_prompt(current_conversation, outline, interviewer_strategy)
                    for current_conversation, outline in zip(current_conversations[start_idx:end_idx], outlines)
                ]

            interviewer_responses = generate_vllm_INTERVIEWER_response_batch(interviewer_prompts, interviewer_model, interviewer_tokenizer)
            interviewer_questions = [
                extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"answer not in brackets:\n {response}"
                for response in interviewer_responses
            ]
            current_conversations[start_idx:end_idx] = [
                f"{ch}\nInterviewer: {question}"
                for ch, question in zip(current_conversations[start_idx:end_idx], interviewer_questions)
            ]

        del interviewer_model
        gc.collect()

        #### Source
        print(f"Turn {turn + 1}: Generating source responses")
        source_model = load_vllm_model(source_model_name)
        source_tokenizer = initialize_tokenizer(source_model_name)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_df = df.iloc[start_idx:end_idx].copy()
            info_items = batch_df['info_items']

            if turn == 0:
                source_prompts = [
                    get_source_starting_prompt(current_conversation, info_item_list)
                    for current_conversation, info_item_list in zip(current_conversations[start_idx:end_idx], info_items)
                ]
            elif turn == num_turns - 1:
                source_prompts = [
                    get_source_ending_prompt(current_conversation, info_item_list)
                    for current_conversation, info_item_list in zip(current_conversations, info_items)
                ]
            else:
                specific_info_item_prompts = [
                    get_source_specific_info_item_prompt(current_conversation, info_item_list)
                    for current_conversation, info_item_list in zip(current_conversations[start_idx:end_idx], info_items)
                ]

                interviewee_specific_item_responses = generate_vllm_SOURCE_response_batch(specific_info_item_prompts, source_model, source_tokenizer)

                specific_info_items = [
                extract_text_inside_brackets(response) if extract_information_item_numbers(extract_text_inside_brackets(response)) else "No information items align with the question"
                for response in interviewee_specific_item_responses
                ]

                for idx, specific_item in enumerate(specific_info_items):
                    info_item_numbers = extract_information_item_numbers(specific_item)
                    unique_info_items_sets[start_idx + idx].update(info_item_numbers)
                
                source_prompts = [
                    get_source_prompt_basic(current_conversation, info_item_list, specific_info_item, "honest")
                    for current_conversation, info_item_list, specific_info_item in zip(current_conversations[start_idx:end_idx], info_items, specific_info_items)
                ]

            source_responses = generate_vllm_SOURCE_response_batch(source_prompts, source_model, source_tokenizer)
            source_answers = [extract_text_inside_brackets(response) for response in source_responses]
            current_conversations[start_idx:end_idx] = [
                f"{ch}\nInterviewee: {response}"
                for ch, response in zip(current_conversations[start_idx:end_idx], source_answers)
            ]

        del source_model
        gc.collect()

    unique_info_item_counts = [len(info_set) for info_set in unique_info_items_sets]

    output_df = pd.DataFrame({
        'id': df['id'],
        'combined_dialogue': df['combined_dialogue'],
        'info_items': df['info_items'],
        'outlines': df['outlines'],
        'final_conversations': current_conversations,
        'total_info_items_extracted': unique_info_item_counts,  # unique info items extracted by interviewer
        'total_info_item_count': total_info_item_counts  # total info items the source has
    })

    output_file_name = os.path.join(output_dir, 'all_basic_interviews_conducted_v2.csv')
    output_df.to_csv(output_file_name, index=False)
    print(f"All interviews saved to {output_file_name}")
    return output_df

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_path, 'news-interview-question-generation')
    dataset_path = os.path.join(project_root, "output_results/game_sim/outlines/final_df_with_outlines.csv")
    df = pd.read_csv(dataset_path)
    df = df.head(4)
    print(df)

    num_turns = 2
    simulated_interviews = conduct_basic_interviews_batch(num_turns, 
                                                          df, 
                                                          interviewer_model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
                                                          source_model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                                                          output_dir="output_results/game_sim/conducted_interviews_basic/interviewer-8B-vs-source-8B")
    print(simulated_interviews)
    print(simulated_interviews['final_conversations'].iloc[0])
    
    # print(f"dataset with simulated interviews: {simulated_interviews}\n")
    # for i, interview in enumerate(simulated_interviews['final_conversations']):
    #     print(f"Interview {i+1}:\n {interview}\n\n\n")

'''
from the dataset of interviews, from each row (interview), plug info_items into source LLM and outlines into interviewer LLM. Then, simulate interview.
column structure of the database outputted:
'id' | 'combined_dialogue' | 'info_items' | 'outlines' | 'final_conversations'
'''

# response = "Information Item #12, information Item 324567, information iTem #696969"
# print(extract_information_item_numbers(response))