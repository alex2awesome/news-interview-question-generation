# conduct_interviews_advanced.py

import os
import sys
import re
import pandas as pd
import random
import ast
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
import numpy as np
from scipy.stats import beta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import load_vllm_model, initialize_tokenizer, extract_text_inside_brackets, stitch_csv_files, find_project_root
from game_sim.game_sim_prompts import get_source_starting_prompt, get_source_ending_prompt, get_source_specific_info_item_prompt, get_source_persuasion_level_prompt, get_advanced_source_persona_prompt, get_advanced_interviewer_prompt, get_interviewer_starting_prompt, get_interviewer_ending_prompt
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

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

# from "Information Item {integer}", extracts {integer}
def extract_information_item_numbers(response):
    return [int(num) for num in re.findall(r'(?i)information item #?(\d+)', response)]

# return total num of matches to "Information item #{integer}"
def count_information_items(info_items_text):
    return len(re.findall(r'(?i)information item #?\d+', info_items_text))

# select random segments from a specified information item based on persona and persuasion level
def get_random_segments(segmented_info_items_str, chosen_info_item, used_segments_dict, persona, persuasion_level, max_proportion=1.0):
    try:
        segmented_info_items = ast.literal_eval(segmented_info_items_str)
        # segmented_info_items = json.loads(sample['segmented_info_items'])
    except:
        return "Error: Unable to parse segmented info items.", used_segments_dict

    match = re.search(r'#(\d+)', chosen_info_item)
    if not match:
        return "Error: Unable to parse chosen_info_item.", used_segments_dict
    item_number = int(match.group(1))
    item_key = f"Information item #{item_number}"

    if item_key not in segmented_info_items:
        return "No segments found for this information item.", used_segments_dict

    segments = segmented_info_items[item_key]
    
    if item_key not in used_segments_dict:
        used_segments_dict[item_key] = set()

    available_segments = [seg for i, seg in enumerate(segments) if i not in used_segments_dict[item_key]]

    if not available_segments:
        return "All segments for this item have been used.", used_segments_dict

    proportion_to_return = sample_proportion_from_beta(persona, persuasion_level)
    proportion_to_return = min(proportion_to_return, max_proportion)

    num_segments_to_return = round(proportion_to_return * len(available_segments))
    num_segments_to_return = max(1, num_segments_to_return)  # Ensure at least one segment is returned

    selected_segments = random.sample(available_segments, num_segments_to_return)

    for seg in selected_segments:
        used_segments_dict[item_key].add(segments.index(seg))

    formatted_segments = "\n".join(f"- {segment}" for segment in selected_segments)

    return formatted_segments, used_segments_dict

# parameters of the beta distribution 
PERSONA_DICT = {
    'anxious': [(2, 4), (4, 4), (4, 2)],
    'avoidant': [(1.5, 3), (4, 3), (7, 2)],
    'straightforward': [(4, 1.5), (5, 1), (7, 0.5)], 
    'poor explainer': [(2, 3), (3, 2), (4, 1)], 
    'dominating': [(5, 1.5), (6, 1), (8, 0.5)], 
    'clueless': [(1, 4), (2, 3), (3, 2)] 
}

def sample_proportion_from_beta(persona, persuasion_level):
    """
    Sample a proportion from the beta distribution based on persona and persuasion level.

    Parameters:
        - persona (str): The persona of the source (e.g., 'anxious', 'avoidant', etc.).
        - persuasion_level (int): The level of persuasion (0, 1, or 2).

    Returns:
        float: A proportion between 0 and 1 sampled from the beta distribution.
    """

    a, b = PERSONA_DICT[persona][persuasion_level]
    proportion = beta.rvs(a, b)
    proportion = max(0.0, min(1.0, proportion))

    return proportion

def conduct_advanced_interviews_batch(num_turns, df, model_name="meta-llama/Meta-Llama-3-70B-Instruct", batch_size=50, output_dir="output_results/game_sim/conducted_interviews_advanced"):
    os.makedirs(output_dir, exist_ok=True)
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    num_samples = len(df)
    unique_info_item_counts = [0] * num_samples
    total_info_item_counts = [0] * num_samples

    persona_types = ["avoidant", "defensive", "straightforward", 
                     "poor explainer", "dominating", "clueless"]

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx]
        info_items = batch_df['info_items']
        outlines = batch_df['outlines']
        current_conversations = [""] * (end_idx - start_idx)
        unique_info_items_sets = [set() for _ in range(end_idx - start_idx)]
        total_info_item_counts[start_idx:end_idx] = [count_information_items(info_item) for info_item in info_items]
        
        total_segments_counts = []
        extracted_segments_sets = [set() for _ in range(end_idx - start_idx)]
        used_segments_dicts = [{} for _ in range(end_idx - start_idx)]
        personas = [random.choice(persona_types) for _ in range(end_idx - start_idx)]
        
        for segmented_items in batch_df['segmented_info_items']:
            segmented_dict = ast.literal_eval(segmented_items)
            total_segments = sum(len(segments) for segments in segmented_dict.values())
            total_segments_counts.append(total_segments)

        #### 1. Handle the FIRST interviewer question and source answer outside the loop
        
        # first interviewer question
        starting_interviewer_prompts = [
            get_interviewer_starting_prompt(outline, "straightforward")
            for outline in outlines
        ]
        starting_interviewer_responses = generate_vllm_INTERVIEWER_response_batch(starting_interviewer_prompts, model, tokenizer)
        starting_interviewer_questions = [
            extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"answer not in brackets:\n {response}"
            for response in starting_interviewer_responses
        ]
        current_conversations = [
            f"{ch}\nInterviewer: {question}"
            for ch, question in zip(current_conversations, starting_interviewer_questions)
        ]

        # first source response
        starting_source_prompts = [
            get_source_starting_prompt(current_conversation, info_item_list, persona)
            for current_conversation, info_item_list, persona in zip(current_conversations, info_items, personas)
        ]
        starting_interviewee_responses = generate_vllm_SOURCE_response_batch(starting_source_prompts, model, tokenizer)
        starting_interviewee_answers = [extract_text_inside_brackets(response) for response in starting_interviewee_responses]
        current_conversations = [
            f"{ch}\nInterviewee: {response}"
            for ch, response in zip(current_conversations, starting_interviewee_answers)
        ]

        #### 2. Handle the middle questions/answers within the loop
        for turn in range(num_turns - 2):
            num_turns_left = num_turns - (2 + turn)
            
            # interviewer question
            interviewer_prompts = [
                get_advanced_interviewer_prompt(current_conversation, outline, num_turns_left, "straightforward")
                for current_conversation, outline in zip(current_conversations, outlines)
            ]
            interviewer_responses = generate_vllm_INTERVIEWER_response_batch(interviewer_prompts, model, tokenizer)
            interviewer_questions = [extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"answer not in brackets:\n {response}" for response in interviewer_responses]
            gc.collect()

            current_conversations = [
                f"{ch}\nInterviewer: {question}"
                for ch, question in zip(current_conversations, interviewer_questions)
            ]

            # get specific information item
            specific_info_item_prompts = [
                get_source_specific_info_item_prompt(current_conversation, info_item_list)
                for current_conversation, info_item_list in zip(current_conversations, info_items)
            ]
            interviewee_specific_item_responses = generate_vllm_SOURCE_response_batch(specific_info_item_prompts, model, tokenizer)
            specific_info_items = [
                extract_text_inside_brackets(response) if extract_information_item_numbers(extract_text_inside_brackets(response)) else "No information items align with the question"
                for response in interviewee_specific_item_responses
            ]

            # get level of persuasion
            persuasion_level_prompts = [
                get_source_persuasion_level_prompt(current_conversation, persona)
                for current_conversation, persona in zip(current_conversations, personas)
            ]
            interviewee_persuasion_level_responses = generate_vllm_SOURCE_response_batch(persuasion_level_prompts, model, tokenizer)
            persuasion_levels = [
                extract_text_inside_brackets(response)
                for response in interviewee_persuasion_level_responses
            ]
            
            # get specific info segments
            random_segments = []
            for idx, (specific_item, segmented_items, persuasion_level_str, persona) in enumerate(zip(specific_info_items, batch_df['segmented_info_items'], persuasion_levels, personas)):
                info_item_numbers = extract_information_item_numbers(specific_item)
                unique_info_items_sets[idx].update(info_item_numbers)
                
                if info_item_numbers:
                    chosen_item = f"Information Item #{info_item_numbers[0]}"

                    try:
                        persuasion_level_int = int(persuasion_level_str)
                    except (ValueError, TypeError):
                        persuasion_level_int = 0
                    if persuasion_level_int not in [0, 1, 2]:
                        persuasion_level_int = 0

                    segments, used_segments_dicts[idx] = get_random_segments(
                        segmented_items, 
                        chosen_item, 
                        used_segments_dicts[idx], 
                        persona,
                        persuasion_level_int)
                    extracted_segments_sets[idx].update([seg.strip() for seg in segments.split('\n') if seg.strip()])
                else:
                    segments = "No specific information item was chosen."
                random_segments.append(segments)

            gc.collect()

            # source response
            source_prompts = [
                get_advanced_source_persona_prompt(current_conversation, info_item_list, random_segment, persona)
                for current_conversation, info_item_list, random_segment, persona in zip(current_conversations, info_items, random_segments, personas)
            ]
            interviewee_responses = generate_vllm_SOURCE_response_batch(source_prompts, model, tokenizer)
            interviewee_answers = [extract_text_inside_brackets(response) for response in interviewee_responses]
            current_conversations = [
                f"{ch}\nInterviewee: {response}"
                for ch, response in zip(current_conversations, interviewee_answers)
            ]

        #### 3. Handle the FINAL interviewer question and source answer outside the loop
        # Last interviewer question (ending prompt)
        interviewer_ending_prompts = [
            get_interviewer_ending_prompt(current_conversation, outline, "straightforward")
            for current_conversation, outline in zip(current_conversations, outlines)
        ]
        ending_interviewer_responses = generate_vllm_INTERVIEWER_response_batch(interviewer_ending_prompts, model, tokenizer)
        ending_interviewer_questions = [
            extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"answer not in brackets:\n {response}"
            for response in ending_interviewer_responses
        ]
        current_conversations = [
            f"{ch}\nInterviewer: {question}"
            for ch, question in zip(current_conversations, ending_interviewer_questions)
        ]

        ending_source_prompts = [
            get_source_ending_prompt(current_conversation, info_item_list, persona)
            for current_conversation, info_item_list, persona in zip(current_conversations, info_items, personas)
        ]
        ending_interviewee_responses = generate_vllm_SOURCE_response_batch(ending_source_prompts, model, tokenizer)
        ending_interviewee_answers = [extract_text_inside_brackets(response) for response in ending_interviewee_responses]
        current_conversations = [
            f"{ch}\nInterviewee: {response}"
            for ch, response in zip(current_conversations, ending_interviewee_answers)
        ]

        unique_info_item_counts[start_idx:end_idx] = [len(info_set) for info_set in unique_info_items_sets]
        extracted_segments_counts = [len(extracted_set) for extracted_set in extracted_segments_sets]

        batch_output_df = pd.DataFrame({
            'id': batch_df['id'],
            'combined_dialogue': batch_df['combined_dialogue'],
            'info_items': batch_df['info_items'],
            'outlines': batch_df['outlines'],
            'final_conversations': current_conversations,
            'total_info_items_extracted': unique_info_item_counts[start_idx:end_idx],
            'total_info_item_count': total_info_item_counts[start_idx:end_idx],
            'extracted_segments_counts': extracted_segments_counts,
            'total_segments_counts': total_segments_counts
        })

        batch_file_name = f"conducted_interviews_batch_{start_idx}_{end_idx}.csv"
        batch_file_path = os.path.join(output_dir, batch_file_name)
        batch_output_df.to_csv(batch_file_path, index=False)
        print(f"Batch {start_idx} to {end_idx} saved to {batch_file_path}")

    final_df = stitch_csv_files(output_dir, 'all_advanced_interviews_conducted.csv')
    return final_df

def conduct_interactive_interview(num_turns, df, model_name="meta-llama/Meta-Llama-3-70B-Instruct", output_dir="output_results/game_sim/conducted_interviews_advanced/human_eval"):
    os.makedirs(output_dir, exist_ok=True)
    role = input("Would you like to play as the interviewer (A) or source (B)? Please type 'A' or 'B': ").upper()
    while role not in ["A", "B"]:
        role = input("Invalid input. Please type 'A' for interviewer or 'B' for source: ").upper()
    role = "interviewer" if role == "A" else "source"
    print(f"\nYou've chosen to play as the {role}.\n")

    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    sample = df.iloc[0] # can change this to randomly sample an interview instead
    info_items = sample['info_items']
    outline = sample['outlines']
    segmented_info_items = ast.literal_eval(sample['segmented_info_items'])

    current_conversation = ""
    unique_info_items_set = set()
    used_segments_dict = {}
    persona_types = ["avoidant", "defensive", "straightforward", "poor explainer", "dominating", "clueless"]
    if role == 'interviewer':
        print("Please choose a source persona to play against.")
    else:
        print("Please choose your source persona.")

    print("Here are your options:")
    for idx, persona_name in enumerate(persona_types):
        print(f"{idx + 1}. {persona_name}")

    user_input = input("Pick a persona (please type a number from 1 to 6): ")

    try:
        index = int(user_input) - 1
        if 0 <= index < len(persona_types):
            persona = persona_types[index]
            print(f"You have selected: {persona}")
        else:
            print("Invalid selection. Randomly picking a persona for you.")
            persona = random.choice(persona_types)
            print(f"Randomly selected persona: {persona}")
    except ValueError:
        print("Invalid input. Randomly picking a persona for you.")
        persona = random.choice(persona_types)
        print(f"Randomly selected persona: {persona}")

    total_info_item_count = count_information_items(info_items)
    total_segments_count = sum(len(segments) for segments in segmented_info_items.values())
    extracted_segments_set = set()

    #### 1. FIRST interviewer question and source answer
    if role == "interviewer":
        interviewer_prompt = get_interviewer_starting_prompt(outline, num_turns, "straightforward")
        print(f"\n{interviewer_prompt}")
        print("No need to wrap your response in brackets, please disregard the last section above")
        human_question = input("Interviewer: ")
        current_conversation = f"(Human) Interviewer: {human_question}"
        
        starting_source_prompt = get_source_starting_prompt(current_conversation, info_items, persona)
        source_response = generate_vllm_SOURCE_response_batch([starting_source_prompt], model, tokenizer)
        source_answer = extract_text_inside_brackets(source_response[0]) or source_response[0]
        current_conversation += f"\nInterviewee: {source_answer}"
        print(f"Interviewee (LLM): {source_answer}")
    else:
        starting_interviewer_prompt = get_interviewer_starting_prompt(outline, num_turns, "straightforward")
        interviewer_response = generate_vllm_INTERVIEWER_response_batch([starting_interviewer_prompt], model, tokenizer)
        interviewer_question = extract_text_inside_brackets(interviewer_response[0]) or interviewer_response[0]
        current_conversation = f"Interviewer: {interviewer_question}"
        print(f"Interviewer (LLM): {interviewer_question}")
        
        source_prompt = get_source_starting_prompt(current_conversation, info_items, persona)
        print(f"\n{source_prompt}")
        print("No need to wrap your response in brackets, please disregard the last section above")
        human_answer = input("Interviewee: ")
        current_conversation += f"\n (Human) Interviewee: {human_answer}"

    #### 2. Middle turns
    for turn in range(num_turns - 2):
        num_turns_left = num_turns - (1 + turn)

        if role == "interviewer":
            interviewer_prompt = get_advanced_interviewer_prompt(current_conversation, outline, num_turns_left, "straightforward")
            print(f"\n{interviewer_prompt}")
            print("No need to wrap your response in brackets, please disregard the last section above")
            human_question = input("\nYour question (Interviewer): ")
            current_conversation += f"\n (Human) Interviewer: {human_question}"

            specific_info_item_prompt = get_source_specific_info_item_prompt(current_conversation, info_items)
            specific_info_item_response = generate_vllm_SOURCE_response_batch([specific_info_item_prompt], model, tokenizer)
            specific_info_item = extract_text_inside_brackets(specific_info_item_response[0]) or "No information items align with the question"

            persuasion_level_prompt = get_source_persuasion_level_prompt(current_conversation, persona)
            persuasion_level_response = generate_vllm_SOURCE_response_batch([persuasion_level_prompt], model, tokenizer)
            persuasion_level = extract_text_inside_brackets(persuasion_level_response[0]) or "0"

            info_item_numbers = extract_information_item_numbers(specific_info_item)
            unique_info_items_set.update(info_item_numbers)
            
            if info_item_numbers:
                chosen_item = f"Information Item #{info_item_numbers[0]}"
                persuasion_level_int = int(persuasion_level) if persuasion_level.isdigit() else 0
                segments, used_segments_dict = get_random_segments(segmented_info_items, chosen_item, used_segments_dict, persona, persuasion_level_int)
                extracted_segments_set.update([seg.strip() for seg in segments.split('\n') if seg.strip()])
            else:
                segments = "No specific information item was chosen."

            source_prompt = get_advanced_source_persona_prompt(current_conversation, info_items, segments, persona)
            source_response = generate_vllm_SOURCE_response_batch([source_prompt], model, tokenizer)
            source_answer = extract_text_inside_brackets(source_response[0]) or source_response[0]
            current_conversation += f"\nInterviewee: {source_answer}"
            print(f"Interviewee (LLM): {source_answer}")
        else:
            interviewer_prompt = get_advanced_interviewer_prompt(current_conversation, outline, num_turns_left, "straightforward")
            interviewer_response = generate_vllm_INTERVIEWER_response_batch([interviewer_prompt], model, tokenizer)
            interviewer_question = extract_text_inside_brackets(interviewer_response[0]) or interviewer_response[0]
            current_conversation += f"\nInterviewer: {interviewer_question}"
            print(f"Interviewer (LLM): {interviewer_question}")

            specific_info_item_prompt = get_source_specific_info_item_prompt(current_conversation, info_items)
            print(specific_info_item_prompt)
            specific_info_item = input("Please pick the most relevant information item: ")
            
            persuasion_level_prompt = get_source_persuasion_level_prompt(current_conversation, persona)
            print(persuasion_level_prompt)
            persuasion_level = input("Please respond with 0, 1, or 2 (measuring persuasiveness): ")

            info_item_numbers = extract_information_item_numbers(specific_info_item)
            unique_info_items_set.update(info_item_numbers)
            persuasion_level_int = int(persuasion_level) if persuasion_level.isdigit() else 0

            if info_item_numbers:
                chosen_item = f"Information Item #{info_item_numbers[0]}"
                segments, used_segments_dict = get_random_segments(segmented_info_items, chosen_item, used_segments_dict, persona, persuasion_level_int)
                extracted_segments_set.update([seg.strip() for seg in segments.split('\n') if seg.strip()])
            else:
                segments = "No specific information item was chosen."

            
            source_prompt = get_advanced_source_persona_prompt(current_conversation, info_items, segments, persona)
            print(f"\n{source_prompt}")
            print("No need to wrap your response in brackets, please disregard the last section above")
            human_answer = input("Your Response to Interviewer's Question: ")
            current_conversation += f"\nInterviewee: {human_answer}"

    #### 3. FINAL interviewer question and source answer
    if role == "interviewer":
        interviewer_prompt = get_interviewer_ending_prompt(current_conversation, outline, "straightforward")
        print(f"\n{interviewer_prompt}")
        print("No need to wrap your response in brackets, please disregard the last section above")
        human_question = input("\nYour final remark (Interviewer): ")
        current_conversation += f"\n (Human) Interviewer: {human_question}"

        source_prompt = get_source_ending_prompt(current_conversation, info_items, persona)
        source_response = generate_vllm_SOURCE_response_batch([source_prompt], model, tokenizer)
        source_answer = extract_text_inside_brackets(source_response[0]) or source_response[0]
        current_conversation += f"\nInterviewee: {source_answer}"
        print(f"Interviewee (LLM): {source_answer}")
    else:
        interviewer_prompt = get_interviewer_ending_prompt(current_conversation, outline, "straightforward")
        interviewer_response = generate_vllm_INTERVIEWER_response_batch([interviewer_prompt], model, tokenizer)
        interviewer_question = extract_text_inside_brackets(interviewer_response[0]) or interviewer_response[0]
        current_conversation += f"\nInterviewer: {interviewer_question}"
        print(f"Interviewer (LLM): {interviewer_question}")

        source_prompt = get_source_ending_prompt(current_conversation, info_items, persona)
        print(f"\n{source_prompt}")
        print("No need to wrap your response in brackets, please disregard the last section above")
        human_answer = input("Interviewee: ")
        current_conversation += f"\n (Human) Interviewee: {human_answer}"

    print("\nFinal Interview Conversation:")
    print(current_conversation)

    output_df = pd.DataFrame({
    'id': sample['id'],
    'combined_dialogue': sample['combined_dialogue'],
    'info_items': sample['info_items'],
    'segmented_info_items': sample['segmented_info_items'],
    'outlines': sample['outlines'],
    'persona chosen': [persona],
    'final_conversation': [current_conversation],
    'total_info_items_extracted': [len(unique_info_items_set)],
    'total_info_item_count': [total_info_item_count],
    'extracted_segments_count': [len(extracted_segments_set)],
    'total_segments_count': [total_segments_count]
    })

    output_path = os.path.join(output_dir, f"human_{role}_vs_LLM_interview_{sample['id']}.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Interview saved to {output_path}")

    return output_df

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_path, 'news-interview-question-generation')
    dataset_path = os.path.join(project_root, "output_results/game_sim/outlines/final_df_with_outlines.csv")
    df = pd.read_csv(dataset_path)
    df = df.head(10)
    print(df)
    # df has columns info_items and outlines
    num_turns = 4
    # simulated_interviews = conduct_advanced_interviews_batch(num_turns, df, model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    
    # print(f"dataset with simulated interviews: {simulated_interviews}\n")
    # for i, interview in enumerate(simulated_interviews['final_conversations']):
    #     print(f"Interview {i+1}:\n {interview}\n\n\n")

    # HUMAN EVAL:
    human_eval = conduct_interactive_interview(num_turns, df, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    print(human_eval)
'''
from the dataset of interviews, from each row (interview), plug info_items into source LLM and outlines into interviewer LLM. Then, simulate interview.
column structure of the database outputted:
'id' | 'combined_dialogue' | 'info_items' | 'outlines' | 'final_conversations'
'''
