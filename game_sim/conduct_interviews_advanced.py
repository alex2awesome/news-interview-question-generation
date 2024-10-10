# conduct_interviews_advanced.py

import os
import sys
import re
import pandas as pd
import random
import json
import ast
from vllm import LLM, SamplingParams
import gc
import numpy as np
from scipy.stats import beta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import (
    load_model, 
    extract_text_inside_brackets, 
    stitch_csv_files, 
    find_project_root,
    generate_INTERVIEWER_response_batch,
    generate_SOURCE_response_batch,
    extract_information_item_numbers,
    count_information_items

)
from game_sim.game_sim_prompts import (
    get_source_starting_prompt, 
    get_source_ending_prompt, 
    get_source_specific_info_items_prompt, 
    get_source_persuasion_level_prompt, 
    get_source_persona_prompt_advanced, 
    get_advanced_interviewer_prompt, 
    get_interviewer_starting_prompt, 
    get_interviewer_ending_prompt
)
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


# select random segments from a specified information item based on persona and persuasion level
# def get_random_segments(segmented_info_items_str, chosen_info_item, used_segments_dict, persona, persuasion_level, max_proportion=1.0):
#     try:
#         segmented_info_items = ast.literal_eval(segmented_info_items_str)
#         # segmented_info_items = json.loads(sample['segmented_info_items'])
#     except:
#         return "Error: Unable to parse segmented info items.", used_segments_dict

#     match = re.search(r'#(\d+)', chosen_info_item)
#     if not match:
#         return "Error: Unable to parse chosen_info_item.", used_segments_dict
#     item_number = int(match.group(1))
#     item_key = f"Information item #{item_number}"

#     if item_key not in segmented_info_items:
#         return "No segments found for this information item.", used_segments_dict

#     segments = segmented_info_items[item_key]
    
#     if item_key not in used_segments_dict:
#         used_segments_dict[item_key] = set()

#     available_segments = [seg for i, seg in enumerate(segments) if i not in used_segments_dict[item_key]]

#     if not available_segments:
#         return "All segments for this item have been used.", used_segments_dict

#     proportion_to_return = sample_proportion_from_beta(persona, persuasion_level)
#     proportion_to_return = min(proportion_to_return, max_proportion)

#     num_segments_to_return = round(proportion_to_return * len(available_segments))
#     num_segments_to_return = max(1, num_segments_to_return)  # Ensure at least one segment is returned

#     selected_segments = random.sample(available_segments, num_segments_to_return)

#     for seg in selected_segments:
#         used_segments_dict[item_key].add(segments.index(seg))

#     formatted_segments = "\n".join(f"- {segment}" for segment in selected_segments)

#     return formatted_segments, used_segments_dict

# parameters of the beta distribution 
PERSONA_DICT = {
    'anxious': [(2, 4), (4, 4), (4, 2)],
    'defensive': [(1.5, 3), (4, 3), (7, 2)],
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

def conduct_advanced_interviews_batch(num_turns, df, model_name="meta-llama/meta-llama-3.1-70b-instruct", batch_size=50, output_dir="output_results/game_sim/conducted_interviews_advanced"):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_name)
    num_samples = len(df)
    unique_info_item_counts = [0] * num_samples
    total_info_item_counts = [0] * num_samples
    df['info_items_dict'] = df['info_items_dict'].apply(json.loads)

    persona_types = ["avoidant", "defensive", "straightforward", 
                     "poor explainer", "dominating", "clueless"]

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx]
        info_items_list = batch_df['info_items']
        info_items_dict = batch_df['info_items_dict']
        outlines = batch_df['outlines']
        current_conversations = [""] * (end_idx - start_idx)
        unique_info_items_sets = [set() for _ in range(end_idx - start_idx)]
        total_info_item_counts[start_idx:end_idx] = [count_information_items(info_items) for info_items in info_items_list]
        
        personas = [random.choice(persona_types) for _ in range(end_idx - start_idx)]

        #### 1. Handle the FIRST interviewer question and source answer outside the loop
        
        # first interviewer question
        starting_interviewer_prompts = [
            get_interviewer_starting_prompt(outline, "straightforward")
            for outline in outlines
        ]
        starting_interviewer_responses = generate_INTERVIEWER_response_batch(starting_interviewer_prompts, model)
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
            get_source_starting_prompt(current_conversation, info_items, persona)
            for current_conversation, info_items, persona in zip(current_conversations, info_items_list, personas)
        ]
        starting_interviewee_responses = generate_SOURCE_response_batch(starting_source_prompts, model)
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
            interviewer_responses = generate_INTERVIEWER_response_batch(interviewer_prompts, model)
            interviewer_questions = [extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"answer not in brackets:\n {response}" for response in interviewer_responses]
            gc.collect()

            current_conversations = [
                f"{ch}\nInterviewer: {question}"
                for ch, question in zip(current_conversations, interviewer_questions)
            ]

            # get specific information item
            specific_info_item_prompts = [
                get_source_specific_info_items_prompt(current_conversation, info_items)
                for current_conversation, info_items in zip(current_conversations, info_items_list)
            ]
            interviewee_specific_item_responses = generate_SOURCE_response_batch(specific_info_item_prompts, model)
            specific_info_items = [
                extract_text_inside_brackets(response) if extract_information_item_numbers(extract_text_inside_brackets(response)) else "No information items align with the question"
                for response in interviewee_specific_item_responses
            ]

            # get level of persuasion
            persuasion_level_prompts = [
                get_source_persuasion_level_prompt(current_conversation, persona)
                for current_conversation, persona in zip(current_conversations, personas)
            ]
            interviewee_persuasion_level_responses = generate_SOURCE_response_batch(persuasion_level_prompts, model)
            persuasion_levels = [
                extract_text_inside_brackets(response)
                for response in interviewee_persuasion_level_responses
            ]
            
            for idx, (specific_item, persuasion_level_str, persona) in enumerate(zip(specific_info_items, persuasion_levels, personas)):
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

            gc.collect()

            # source response
            source_prompts = [
                get_source_persona_prompt_advanced(current_conversation, info_items, persona, )
                for current_conversation, info_items, persona in zip(current_conversations, info_items_list, personas)
            ]
            interviewee_responses = generate_SOURCE_response_batch(source_prompts, model)
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
        ending_interviewer_responses = generate_INTERVIEWER_response_batch(interviewer_ending_prompts, model)
        ending_interviewer_questions = [
            extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"answer not in brackets:\n {response}"
            for response in ending_interviewer_responses
        ]
        current_conversations = [
            f"{ch}\nInterviewer: {question}"
            for ch, question in zip(current_conversations, ending_interviewer_questions)
        ]

        ending_source_prompts = [
            get_source_ending_prompt(current_conversation, info_items, persona)
            for current_conversation, info_items, persona in zip(current_conversations, info_items_list, personas)
        ]
        ending_interviewee_responses = generate_SOURCE_response_batch(ending_source_prompts, model)
        ending_interviewee_answers = [extract_text_inside_brackets(response) for response in ending_interviewee_responses]
        current_conversations = [
            f"{ch}\nInterviewee: {response}"
            for ch, response in zip(current_conversations, ending_interviewee_answers)
        ]

        unique_info_item_counts[start_idx:end_idx] = [len(info_set) for info_set in unique_info_items_sets]

        batch_output_df = pd.DataFrame({
            'id': batch_df['id'],
            'combined_dialogue': batch_df['combined_dialogue'],
            'info_items': batch_df['info_items'],
            'outlines': batch_df['outlines'],
            'final_conversations': current_conversations,
            'total_info_items_extracted': unique_info_item_counts[start_idx:end_idx],
            'total_info_item_count': total_info_item_counts[start_idx:end_idx],
        })

        batch_file_name = f"conducted_interviews_batch_{start_idx}_{end_idx}.csv"
        batch_file_path = os.path.join(output_dir, batch_file_name)
        batch_output_df.to_csv(batch_file_path, index=False)
        print(f"Batch {start_idx} to {end_idx} saved to {batch_file_path}")

    final_df = stitch_csv_files(output_dir, 'all_advanced_interviews_conducted.csv')
    return final_df

def get_relevant_info_items(info_item_numbers, info_items_dict, persona, persuasion_level, used_info_items):
    available_items = []
    for num in info_item_numbers:
        if num not in used_info_items:
            key = f"information item #{num}".lower()
            content = info_items_dict.get(key, '')
            if content:
                available_items.append((num, f"Information Item #{num}: {content}"))
    N = len(info_item_numbers)
    k = len([num for num in info_item_numbers if num in used_info_items]) # num of used items
    available = N - k

    if available == 0:
        return "All relevant information has been given to the interviewer already!", []  
    if not available_items:
        return "No information items align with the question"
    proportion = sample_proportion_from_beta(persona, persuasion_level)
    num_items = int(proportion * N)
    if num_items > available:
        num_items = available

    if num_items > 0:
        sampled = random.sample(available_items, num_items)
        sampled_items_str = "\n".join(item[1] for item in sampled)
        sampled_item_numbers = [item[0] for item in sampled]
        return sampled_items_str, sampled_item_numbers
    elif int(proportion * N) == 0:
        return "No relevant information for this question. Feel free to ramble and say nothing important.", []
    else:
        return "No relevant information for this question. Feel free to ramble and say nothing important.", []

def get_all_relevant_info_items(info_item_numbers, info_items_dict):
    all_items = []
    for num in info_item_numbers:
        key = f"information item #{num}".lower()
        content = info_items_dict.get(key, '')
        if content:
            all_items.append(f"Information Item #{num}: {content}")

    if not all_items:
        return "No information items align with the question"

    return "\n".join(all_items) if all_items else "No information items align with the question"

# ANSI escape codes for colors
RESET = "\033[0m"        # Resets the color
INTERVIEWER_COLOR = "\033[94m"   # Blue
SOURCE_COLOR = "\033[92m"        # Green
PROMPT_COLOR = "\033[35m"        # Yellow
ERROR_COLOR = "\033[91m"         # Red

def human_eval(
        num_turns,
        df, 
        model_name="meta-llama/meta-llama-3.1-70b-instruct", 
        output_dir="output_results/game_sim/conducted_interviews_advanced/human_eval"
):
    os.makedirs(output_dir, exist_ok=True)
    
    role = input(f"{PROMPT_COLOR}Would you like to play as the interviewer (A) or source (B)? Please type 'A' or 'B': {RESET}").upper()
    while role not in ["A", "B"]:
        role = input(f"{ERROR_COLOR}Invalid input. Please type 'A' for interviewer or 'B' for source: {RESET}").upper()
    role = "interviewer" if role == "A" else "source"
    print(f"\n{PROMPT_COLOR}You've chosen to play as the {role}.{RESET}\n")

    persona_types = ["avoidant", "defensive", "straightforward", "poor explainer", "dominating", "clueless"]
    if role == 'interviewer':
        print(f"{PROMPT_COLOR}Please choose a source persona to play against.{RESET}")
    else:
        print(f"{PROMPT_COLOR}Please choose your source persona.{RESET}")

    print(f"{PROMPT_COLOR}Here are your options:{RESET}")
    for idx, persona_name in enumerate(persona_types):
        print(f"{PROMPT_COLOR}{idx + 1}. {persona_name}{RESET}")

    user_input = input(f"{PROMPT_COLOR}Pick a persona (please type a number from 1 to 6): {RESET}")

    try:
        index = int(user_input) - 1
        if 0 <= index < len(persona_types):
            persona = persona_types[index]
            print(f"\n{PROMPT_COLOR}You have selected: {persona}{RESET}")
        else:
            print(f"{ERROR_COLOR}Invalid selection. Randomly picking a persona for you.{RESET}")
            persona = random.choice(persona_types)
            print(f"{PROMPT_COLOR}Randomly selected persona: {persona}{RESET}")
    except ValueError:
        print(f"{ERROR_COLOR}Invalid input. Randomly picking a persona for you.{RESET}")
        persona = random.choice(persona_types)
        print(f"{PROMPT_COLOR}Randomly selected persona: {persona}{RESET}")

    used_ids = set()
    interview_id_pattern = re.compile(r"^interview_(\d+)_human_[AB]_vs_LLM\.csv$")

    for filename in os.listdir(output_dir):
        match = interview_id_pattern.match(filename)
        if match:
            used_id = match.group(1)
            used_ids.add(used_id)
    available_df = df[~df['id'].astype(str).isin(used_ids)].copy()

    if available_df.empty:
        print(f"{ERROR_COLOR}All interviews from this dataframe have been conducted. No more unique interviews available.{RESET}")
        return None 
    sample = available_df.sample(n=1).iloc[0].copy()
    sample['info_items_dict'] = ast.literal_eval(sample['info_items_dict'])
    info_items = sample['info_items']
    outline = sample['outlines']

    current_conversation = ""
    unique_info_items_set = set()
    total_info_item_count = count_information_items(info_items)

    model = load_model(model_name)

    #### 1. FIRST interviewer question and source answer
    if role == "interviewer":  # human is interviewer
        interviewer_prompt = f'''
        {PROMPT_COLOR}It's the beginning of the interview.

        Here is the outline of objectives you've prepared before the interview:

        {outline}

        Use this outline to help you extract as much information from the source as possible, and think about relevant follow-up questions.{RESET}
        '''
        print(f"\n{interviewer_prompt}")
        human_question = input(f"\n\n{INTERVIEWER_COLOR}You only have time for {num_turns} questions. Now, please input your starting remark: {RESET}")
        current_conversation = f"(Human) Interviewer: {human_question}"

        starting_source_prompt = get_source_starting_prompt(current_conversation, info_items, persona)
        source_response = generate_SOURCE_response_batch([starting_source_prompt], model)
        source_answer = extract_text_inside_brackets(source_response[0]) or source_response[0]
        current_conversation += f"\nInterviewee: {source_answer}"
        print(f"\n{SOURCE_COLOR}Interviewee (LLM): {source_answer}{RESET}\n")
    else:  # human is source
        starting_interviewer_prompt = get_interviewer_starting_prompt(outline, num_turns, "straightforward")
        interviewer_response = generate_INTERVIEWER_response_batch([starting_interviewer_prompt], model)
        interviewer_question = extract_text_inside_brackets(interviewer_response[0]) or interviewer_response[0]
        current_conversation = f"Interviewer: {interviewer_question}"
        print(f"\n{INTERVIEWER_COLOR}Interviewer Starting Remark (LLM): {interviewer_question}{RESET}\n")

        print(f"\n\n{PROMPT_COLOR}Here are all the information items you have in this interview. These represent the relevant information at your disposal to divulge to the interviewer: \n\n{info_items}{RESET}")
        
        human_answer = input(f"\n\n{SOURCE_COLOR}Please respond accordingly: {RESET}")
        current_conversation += f"\n(Human) Interviewee: {human_answer}"

    #### 2. Middle turns
    for turn in range(num_turns - 2):
        num_turns_left = num_turns - (1 + turn)

        if role == "interviewer":  # human is interviewer
            interviewer_prompt = f'''
            {PROMPT_COLOR}Assess whether your previous question was fully answered and whether you can move on to the next one.
            Analyze the source's most recent response and identify their likely emotional/cognitive state (and persona).
            Based on the detected persona, decide how to proceed with your questioning.
            Now, formulate a question that will best guide the source based on their current persona.{RESET}
            '''
            print(f"\n{interviewer_prompt}\n")
            human_question = input(f"\n\n{INTERVIEWER_COLOR}You have {num_turns_left} questions left. Please ask your next question: {RESET}")
            current_conversation += f"\n(Human) Interviewer: {human_question}"

            specific_info_items_prompt = get_source_specific_info_items_prompt(current_conversation, info_items)
            specific_info_items_response = generate_SOURCE_response_batch([specific_info_items_prompt], model)
            all_relevant_info_items = extract_text_inside_brackets(specific_info_items_response[0]) or "No information items align with the question"

            persuasion_level_prompt = get_source_persuasion_level_prompt(current_conversation, persona)
            persuasion_level_response = generate_SOURCE_response_batch([persuasion_level_prompt], model)
            persuasion_level = extract_text_inside_brackets(persuasion_level_response[0]) or "0"

            info_item_numbers = extract_information_item_numbers(all_relevant_info_items)
            persuasion_level_int = int(persuasion_level) if persuasion_level.isdigit() else 0
            relevant_info_items_str, sampled_item_numbers = get_relevant_info_items(
                info_item_numbers, 
                sample['info_items_dict'], 
                persona, 
                persuasion_level_int, 
                unique_info_items_set
            )
            unique_info_items_set.update(sampled_item_numbers)

            source_prompt = get_source_persona_prompt_advanced(current_conversation, relevant_info_items_str, persona, persuasion_level_int)
            source_response = generate_SOURCE_response_batch([source_prompt], model)
            source_answer = extract_text_inside_brackets(source_response[0]) or source_response[0]
            current_conversation += f"\nInterviewee: {source_answer}"
            print(f"\n{SOURCE_COLOR}Interviewee (LLM): {source_answer}{RESET}")
        else:  # human is source
            interviewer_prompt = get_advanced_interviewer_prompt(current_conversation, outline, num_turns_left, "straightforward")
            interviewer_response = generate_INTERVIEWER_response_batch([interviewer_prompt], model)
            interviewer_question = extract_text_inside_brackets(interviewer_response[0]) or interviewer_response[0]
            current_conversation += f"\nInterviewer: {interviewer_question}"
            print(f"\n{INTERVIEWER_COLOR}Interviewer (LLM): {interviewer_question}{RESET}\n\n")

            specific_info_item_prompt = get_source_specific_info_items_prompt(current_conversation, info_items)
            interviewee_specific_item_responses = generate_SOURCE_response_batch([specific_info_item_prompt], model)
            all_relevant_info_items = extract_text_inside_brackets(interviewee_specific_item_responses[0]) if extract_information_item_numbers(extract_text_inside_brackets(interviewee_specific_item_responses[0])) else "No information items align with the question"
            info_item_numbers = extract_information_item_numbers(all_relevant_info_items)  # list of numbers
            
            human_persuation_criteria = f'''
            {PROMPT_COLOR}- 0: The question is not persuasive at all and does nothing to help you trust them more.
            - 1: The question is slightly persuasive and you are be willing to engage more.
            - 2: The question is significantly persuasive and you are fully willing to engage and trust them.{RESET}
            '''
            print(f"\nPlease analyze the interviewer's last question. Given that you are a {persona} source, do you feel persuaded?\n\nEvaluate on the following criteria: \n\n{human_persuation_criteria}\n\n")
            persuasion_level = input(f"\n{PROMPT_COLOR}Now, please respond with either 0, 1, or 2: {RESET}")
            persuasion_level_int = int(persuasion_level) if persuasion_level.isdigit() else 0
            
            sampled_relevant_info_items_str, sampled_item_numbers = get_relevant_info_items(
                info_item_numbers, 
                sample['info_items_dict'], 
                persona, 
                persuasion_level_int, 
                unique_info_items_set
            )
            unique_info_items_set.update(sampled_item_numbers)

            human_source_prompt = f'''
            {PROMPT_COLOR}Here is the list of relevant information items:
                
            {get_all_relevant_info_items(info_item_numbers, sample['info_items_dict'])}

            From this list, we suggest you use the following information items (but you can choose others if you'd like):

            {sampled_relevant_info_items_str}{RESET}
            '''
            print(human_source_prompt)
            human_specific_info_items = input(f"\n\n{SOURCE_COLOR}From the list of relevant information items, please input the ones you'll be using in your response to the interviewer.\nPlease list just the number of the information item(s).\n(If there is more than one, please separate them by commas e.g. 1, 2, 4, 5):{RESET}")
            try:
                human_specific_info_item_numbers = [int(x.strip()) for x in human_specific_info_items.split(',') if x.strip().isdigit()]
                unique_info_items_set.update(human_specific_info_item_numbers)
            except ValueError:
                print(f"{ERROR_COLOR}Invalid input. No information items will be added.{RESET}")

            human_answer = input(f"{SOURCE_COLOR}Now, please respond to the interviewer's question: {RESET}")
            current_conversation += f"\n(Human) Interviewee: {human_answer}"

    #### 3. FINAL interviewer question and source answer
    if role == "interviewer":
        human_question = input(f"\n\n{INTERVIEWER_COLOR}It's time to end this interview. Please input your final remark (Interviewer): {RESET}")
        current_conversation += f"\n(Human) Interviewer: {human_question}"

        source_prompt = get_source_ending_prompt(current_conversation, persona)
        source_response = generate_SOURCE_response_batch([source_prompt], model)
        source_answer = extract_text_inside_brackets(source_response[0]) or source_response[0]
        current_conversation += f"\nInterviewee: {source_answer}"
        print(f"{SOURCE_COLOR}Interviewee (LLM): {source_answer}{RESET}")
    else:
        interviewer_prompt = get_interviewer_ending_prompt(current_conversation, outline, "straightforward")
        interviewer_response = generate_INTERVIEWER_response_batch([interviewer_prompt], model)
        interviewer_question = extract_text_inside_brackets(interviewer_response[0]) or interviewer_response[0]
        current_conversation += f"\nInterviewer: {interviewer_question}"
        print(f"\n{INTERVIEWER_COLOR}Interviewer (LLM): {interviewer_question}{RESET}\n")

        human_answer = input(f"{SOURCE_COLOR}Please input your final ending remark: {RESET}")
        current_conversation += f"\n(Human) Interviewee: {human_answer}"

    # print report
    if role == "interviewer":
        print(f"\n{PROMPT_COLOR}Congratulations! You have successfully conducted an interview as the interviewer.{RESET}")
        print(f"\n{PROMPT_COLOR}You have extracted {len(unique_info_items_set)} unique information items out of {total_info_item_count} total information items.{RESET}")
        print(f"\n{PROMPT_COLOR}Here are the total information items in the interview:\n{info_items}{RESET}")
        print(f"\n{PROMPT_COLOR}You extracted these: {unique_info_items_set}{RESET}")

    output_df = pd.DataFrame({
        'id': sample['id'],
        'combined_dialogue': sample['combined_dialogue'],
        'info_items': sample['info_items'],
        'outlines': sample['outlines'],
        'persona chosen': [persona],
        'final_conversation': [current_conversation],
        'info_item_numbers_used': [unique_info_items_set],
        'total_info_items_extracted': [len(unique_info_items_set)],
        'total_info_item_count': [total_info_item_count],
    })

    output_path = os.path.join(output_dir, f"interview_{sample['id']}_human_{role}_vs_LLM.csv")
    output_df.to_csv(output_path, index=False)
    print(f"{PROMPT_COLOR}Interview saved to {output_path}{RESET}")

    return output_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_turns", type=int, default=4, help="Number of turns in the interview")
    parser.add_argument("--model_name", type=str, default="meta-llama/meta-llama-3.1-70b-instruct", help="Model name")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for conducting interviews")
    parser.add_argument("--dataset_path", type=str, default="output_results/game_sim/outlines/final_df_with_outlines.csv", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="output_results/game_sim/conducted_interviews_advanced", help="Output directory for saving conducted interviews")
    parser.add_argument("--human_eval", action="store_true", help="Conduct human evaluation")

    args = parser.parse_args()

    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_path, 'news-interview-question-generation')
    dataset_path = os.path.join(project_root, args.dataset_path)
    df = pd.read_csv(dataset_path)
    df = df.head(args.batch_size)

    if args.human_eval:
        human_evaluation = human_eval(args.num_turns, df, model_name=args.model_name, output_dir=args.output_dir)
        print(human_evaluation)

    # simulated_interviews = conduct_advanced_interviews_batch(num_turns, df, model_name="meta-llama/meta-llama-3.1-70b-instruct")
    # print(f"dataset with simulated interviews: {simulated_interviews}\n")
    # for i, interview in enumerate(simulated_interviews['final_conversations']):
    #     print(f"Interview {i+1}:\n {interview}\n\n\n")

'''
from the dataset of interviews, from each row (interview), plug info_items into source LLM and outlines into interviewer LLM. Then, simulate interview.
column structure of the database outputted:
'id' | 'combined_dialogue' | 'info_items' | 'outlines' | 'final_conversations'
'''
