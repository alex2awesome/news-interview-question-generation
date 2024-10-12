# conduct_interviews_advanced.py

import os
import sys
import re
import pandas as pd
import random
import json
import ast
import gc
import numpy as np
from tqdm.auto import tqdm
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
    get_source_prompt_basic,
    get_source_prompt_intermediate,
    get_source_persona_prompt_advanced, 
    get_interviewer_prompt,
    get_advanced_interviewer_prompt, 
    get_interviewer_starting_prompt, 
    get_interviewer_ending_prompt
)
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


# parameters of the beta distribution 
PERSONA_DICT = {
    'anxious': [(3, 7), (4, 6), (5, 5), (7, 3.5), (9, 2)],
    'avoidant': [(2.5, 6.5), (5, 6.75), (7, 7), (7.25, 4), (7.5, 1.5)],
    'adversarial': [(1, 9), (2, 8.5), (4, 8), (8, 7.5), (9, 7)],
    'defensive': [(4, 8), (6, 7), (8.5, 6.5), (8.5, 4.25), (8.5, 2)],
    'straightforward': [(2, 5.5), (4, 5.5), (5.5, 5.5), (7.75, 4), (10, 2.5)],
    'poor explainer': [(3.5, 7.5), (5.5, 7.5), (7.5, 7.5), (7.25, 4.35), (7, 1.2)],
    'dominating': [(2, 6), (4, 6), (6, 6), (7.5, 3.8), (8, 1.6)],
    'clueless': [(3.6, 8.0), (4.8, 6.5), (5, 5), (6.55, 3.3), (8.1, 1.6)]
}
UNIFORM_BETA_PARAMS = (1, 1)

def sample_proportion_from_beta(persona, persuasion_level, game_level="advanced"):
    """
    Sample a proportion from the beta distribution based on persona and persuasion level.

    Parameters:
        - persona (str): The persona of the source (e.g., 'anxious', 'dominant', etc.).
        - persuasion_level (int): The level of persuasion (1-5).

    Returns:
        float: A proportion between 0 and 1 sampled from the beta distribution.
    """
    if game_level == "basic":
        return 1 
    elif game_level == "intermediate":
        return beta.rvs(UNIFORM_BETA_PARAMS[0], UNIFORM_BETA_PARAMS[1])
    else:
        a, b = PERSONA_DICT[persona][persuasion_level - 1]
        proportion = beta.rvs(a, b)
        proportion = max(0.0, min(1.0, proportion))
        return proportion


def get_relevant_info_items(info_item_numbers, info_items_dict, persona, persuasion_level, used_info_items, game_level="advanced"):
    """
    Retrieve relevant information items based on the given parameters.

    This function identifies and returns a subset of information items that are relevant to the current 
    interview context. It considers the persona and persuasion level to determine the proportion of 
    information items to return. The function also ensures that previously used information items are not 
    selected again.

    Parameters:
        - info_item_numbers (list of int): A list of information item numbers that are relevant to a previously asked question.
        - info_items_dict (dict): A dictionary mapping all information item keys to their content.
        - persona (str): The persona of the source (e.g., 'anxious', 'dominant', etc.).
        - persuasion_level (int): The level of persuasion (0, 1, or 2).
        - used_info_items (set of int): A set of information item numbers that have already been used.

    Returns:
        tuple: A tuple containing:
            - str: A formatted string of the selected information items or a message if no items are available.
            - list of int: A list of numbers representing the selected information items.
    """
    # Initialize a list to store available information items
    available_items = []
    
    # Iterate over the provided information item numbers
    for num in info_item_numbers:
        # Check if the item has not been used
        if num not in used_info_items:
            # Construct the key for the information item
            key = f"Information Item #{num}"
            # Retrieve the content of the information item from the dictionary
            content = info_items_dict.get(key.lower(), '')
            # If content exists, add it to the available items list
            if content:
                available_items.append((num, f"{key}: {content}"))
    
    # Calculate the total number of items and the number of used items
    N = len(info_item_numbers)
    k = len([num for num in info_item_numbers if num in used_info_items])  # num of used items
    available = N - k  # Calculate the number of available items

    # Check if all relevant information has been used
    if available == 0:
        return "All relevant information has been given to the interviewer already!", []  
    
    # Check if there are no available items
    if not available_items:
        return "No information items align with the question", []
    
    # Sample a proportion of items to return based on persona and persuasion level
    proportion = sample_proportion_from_beta(persona, persuasion_level, game_level=game_level)
    num_items_to_retrieve = int(proportion * N)
    
    # Adjust the number of items to return if it exceeds available items
    num_items_to_retrieve = min(num_items_to_retrieve, available)

    # If there are items to return, sample and format them
    if num_items_to_retrieve > 0:
        sampled = random.sample(available_items, num_items_to_retrieve)
        sampled_items_str = "\n".join(item[1] for item in sampled)
        sampled_item_numbers = [item[0] for item in sampled]
        return sampled_items_str, sampled_item_numbers
    
    # If no items are selected, return a default message
    else:
        return "No relevant information for this question. Feel free to ramble and say nothing important.", []


def get_all_relevant_info_items(info_item_numbers, info_items_dict):
    """
    Retrieve all relevant information items based on provided item numbers.

    This function takes a list of information item numbers and a dictionary containing
    information items. It constructs a list of all relevant information items that
    correspond to the provided item numbers.

    Parameters:
    - info_item_numbers (list of int): A list of numbers representing the information items to retrieve.
    - info_items_dict (dict): A dictionary where keys are information item identifiers and values are the content.

    Returns:
    - str: A formatted string containing all relevant information items. If no items align with the question,
            a default message is returned.
    """
    # Initialize a list to store all relevant information items
    all_items = []
    
    # Iterate over the provided information item numbers
    for num in info_item_numbers:
        # Construct the key for the information item
        key = f"information item #{num}".lower()
        # Retrieve the content of the information item from the dictionary
        content = info_items_dict.get(key, '')
        # If content exists, add it to the all items list
        if content:
            all_items.append(f"Information Item #{num}: {content}")

    # Check if no relevant information items were found
    if not all_items:
        return "No information items align with the question"

    # Return a formatted string of all relevant information items
    return "\n".join(all_items) if all_items else "No information items align with the question"


## Conduct advanced interviews
def conduct_advanced_interviews_batch(
    num_turns, df, 
    interviewer_model_name="meta-llama/meta-llama-3.1-70b-instruct", 
    source_model_name="gpt-4o",
    batch_size=50, 
    output_dir="output_results/game_sim/conducted_interviews_advanced",
    game_level="advanced",
    interviewer_strategy="straightforward"
):
    os.makedirs(output_dir, exist_ok=True)
    interviewer_model = load_model(interviewer_model_name)
    if source_model_name == interviewer_model_name:
        source_model = interviewer_model
    else:
        assert source_model_name == "gpt-4o", "Only GPT-4o is supported as the source model, if it's different from the interviewer model."
        source_model = load_model(source_model_name)

    num_samples = len(df)
    unique_info_item_counts = [0] * num_samples
    total_info_item_counts = [0] * num_samples
    used_info_items = [set() for _ in range(num_samples)]
    df['info_items_dict'] = df['info_items_dict'].apply(json.loads)

    persona_types = [
        "anxious",
        "avoidant",
        "adversarial",
        "defensive", 
        "straightforward", 
        "poor explainer", 
        "dominating", 
        "clueless"
    ]

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx]
        info_items_list = batch_df['info_items']
        outlines = batch_df['outlines']
        current_conversations = [""] * (end_idx - start_idx)
        previous_persuasion_scores_all_rounds = [[]] * (end_idx - start_idx)

        unique_info_items_sets = [set() for _ in range(end_idx - start_idx)]
        total_info_item_counts[start_idx:end_idx] = [count_information_items(info_items) for info_items in info_items_list]
        
        # basic just uses the "straightforward" persona
        if game_level == "basic":
            personas = ["straightforward"] * (end_idx - start_idx)
        else:
            personas = [random.choice(persona_types) for _ in range(end_idx - start_idx)]

        #### 1. Handle the FIRST interviewer question and source answer outside the loop
        # The following section initializes the first question from the interviewer
        # and the first response from the source in the interview process.
        
        # Generate starting prompts for the interviewer using the outline and a straightforward persona
        starting_interviewer_prompts = [
            get_interviewer_starting_prompt(outline, "straightforward")
            for outline in outlines
        ]
        
        # Generate responses for the starting prompts using the interviewer model
        starting_interviewer_responses = generate_INTERVIEWER_response_batch(starting_interviewer_prompts, interviewer_model)
        
        # Extract the questions from the responses, ensuring they are within brackets
        starting_interviewer_questions = [
            extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"answer not in brackets:\n {response}"
            for response in starting_interviewer_responses
        ]
        
        # Update the current conversation with the interviewer's question
        current_conversations = [
            f"{ch}\nInterviewer: {question}"
            for ch, question in zip(current_conversations, starting_interviewer_questions)
        ]
        
        # Generate starting prompts for the source using the current conversation and persona
        starting_source_prompts = [
            get_source_starting_prompt(current_conversation, persona)
            for current_conversation, persona in zip(current_conversations, personas)
        ]
        
        # Generate responses for the starting prompts using the source model
        starting_interviewee_responses = generate_SOURCE_response_batch(starting_source_prompts, source_model)
        
        # Extract the answers from the responses, ensuring they are within brackets
        starting_interviewee_answers = [extract_text_inside_brackets(response) for response in starting_interviewee_responses]
        
        # Update the current conversation with the interviewee's answer
        current_conversations = [
            f"{ch}\nInterviewee: {response}"
            for ch, response in zip(current_conversations, starting_interviewee_answers)
        ]
        
        # Initialize a cache for running conversations, storing the first question and answer
        running_conversations_for_caching = [
            {
                'question_0': question,
                'persuasion_0': None,
                'info_items_0': None,
                'answer_0': answer,
            }
            for question, answer in zip(
                starting_interviewer_questions, 
                starting_interviewee_answers, 
            )
        ]

        #### 2. Handle the middle questions/answers within the loop
        for turn in tqdm(range(num_turns - 1), desc="Conducting interviews", total=num_turns - 1):
            # The following code block handles the middle questions and answers in the interview loop.
            # It iterates over the number of turns left and generates interviewer questions and interviewee responses.

            # Calculate the number of turns left in the conversation
            num_turns_left = num_turns - turn
            
            # Generate prompts for the interviewer based on the current conversation, outline, and remaining turns
            if interviewer_strategy == "straightforward":
                interviewer_prompts = [
                    get_interviewer_prompt(current_conversation, outline, num_turns_left, "straightforward")
                        for current_conversation, outline in zip(current_conversations, outlines)
                ]
            else:
                interviewer_prompts = [
                    get_advanced_interviewer_prompt(current_conversation, outline, num_turns_left, "straightforward")
                        for current_conversation, outline in zip(current_conversations, outlines)
                ]
            
            # Generate responses for the interviewer prompts using the interviewer model
            interviewer_responses = generate_INTERVIEWER_response_batch(interviewer_prompts, interviewer_model)
            
            # Extract questions from the interviewer responses, handling cases where the answer is not in brackets
            interviewer_questions = [
                extract_text_inside_brackets(response) 
                if extract_text_inside_brackets(response) 
                else f"answer not in brackets:\n {response}"
                for response in interviewer_responses
            ]

            # Update the current conversations with the new interviewer questions
            current_conversations = [
                f"{ch}\nInterviewer: {question}"
                for ch, question in zip(current_conversations, interviewer_questions)
            ]

            ## Goal: Find which information items from the interviewee are relevant to the question.
            # Generate prompts 
            specific_info_item_prompts = [
                get_source_specific_info_items_prompt(info_items, final_question)
                for info_items, final_question in zip(info_items_list, interviewer_questions)
            ]
            
            # Generate responses for the specific information item prompts using the source model
            interviewee_specific_item_responses = generate_SOURCE_response_batch(specific_info_item_prompts, source_model)
            
            # Extract specific information items from the responses, handling cases where no items align with the question
            specific_info_items = [
                extract_text_inside_brackets(response)
                for response in interviewee_specific_item_responses
            ]
            specific_info_items_numbers = [
                extract_information_item_numbers(response)
                for response in specific_info_items
            ]

            ## Persuasion level is only used in the advanced game level.
            if game_level == "advanced":
                ## Goal: Find level of persuasion from the interviewee based on the current conversation and persona.
                # Generate prompts
                persuasion_level_prompts = [
                    get_source_persuasion_level_prompt(current_conversation, persona, previous_persuasion_scores)
                    for current_conversation, persona, previous_persuasion_scores in zip(current_conversations, personas, previous_persuasion_scores_all_rounds)
                ]
                
                # Generate responses
                interviewee_persuasion_level_responses = generate_SOURCE_response_batch(persuasion_level_prompts, source_model)
                
                # Extract persuasion levels from the responses
                persuasion_levels = [
                    extract_text_inside_brackets(response)
                    for response in interviewee_persuasion_level_responses
                ]

                # Convert persuasion levels to integers if possible
                persuasion_level_ints = [
                    int(persuasion_level) 
                    if persuasion_level.isdigit() else 0 
                    for persuasion_level in persuasion_levels
                ]

                # Update the previous persuasion scores for the next turn
                previous_persuasion_scores_all_rounds = [
                    previous_persuasion_scores + [persuasion_level_int]
                    for previous_persuasion_scores, persuasion_level_int in zip(previous_persuasion_scores_all_rounds, persuasion_level_ints)
                ]

            else:
                persuasion_levels = ["3"] * len(current_conversations)
                persuasion_level_ints = [3] * len(current_conversations)

            # Filter the specific_info_items to the ones we haven't used yet
            info_items_to_use = [
                get_relevant_info_items(
                    info_item_numbers, 
                    info_items_dict,
                    persona, 
                    persuasion_level, 
                    used,
                    game_level=game_level
                )
                for info_item_numbers, info_items_dict, persona, persuasion_level, used in zip(
                    specific_info_items_numbers, batch_df['info_items_dict'], personas, persuasion_level_ints, used_info_items
                )
            ]

            # Update the used information items
            used_info_items = [
                used.union(set(info_item_numbers[1])) 
                    for used, info_item_numbers in zip(used_info_items, info_items_to_use)
            ]

            # source response:
            # The following section generates prompts for the source's response based on the current conversation, 
            # relevant information items, and the persona of the interviewee.
            # It uses the `get_source_persona_prompt_advanced` function to create these prompts.
            if game_level == "basic":
                source_prompts = [
                    get_source_prompt_basic(current_conversation, info_items[0]) # info_items is a tuple (<concatted info items>, <list of info item numbers>), we just want the first, a string
                        for current_conversation, info_items in zip(current_conversations, info_items_to_use)
                ]
            elif game_level == "intermediate":
                source_prompts = [
                    get_source_prompt_intermediate(current_conversation, info_items[0], persona) # info_items is a tuple (<concatted info items>, <list of info item numbers>), we just want the first, a string
                        for current_conversation, info_items, persona in zip(current_conversations, info_items_to_use, personas)
                ]
            else:
                source_prompts = [
                    get_source_persona_prompt_advanced(current_conversation, info_items[0], persona, persuasion_level) # info_items is a tuple (<concatted info items>, <list of info item numbers>), we just want the first, a string
                        for current_conversation, info_items, persona, persuasion_level 
                        in zip(current_conversations, info_items_to_use, personas, persuasion_level_ints)
                ]
            
            # The generated prompts are then passed to the `generate_SOURCE_response_batch` function to obtain responses
            # from the source model. These responses simulate the interviewee's answers.
            interviewee_responses = generate_SOURCE_response_batch(source_prompts, source_model)
            
            # The responses are processed to extract the text inside brackets, which is considered the final answer.
            interviewee_answers = [
                extract_text_inside_brackets(response) for response in interviewee_responses
            ]
            
            # The current conversation is updated by appending the interviewee's answer to it.
            current_conversations = [
                f"{ch}\nInterviewee: {response}"
                for ch, response in zip(current_conversations, interviewee_answers)
            ]
            
            # The running conversations are updated for caching purposes, including the question, answer, and persuasion level.
            running_conversations_for_caching = [
                {
                    **conversation,
                    f'question_{turn + 1}': question,
                    f'persuasion_{turn + 1}': persuasion,
                    f'info_items_{turn + 1}': info_items,
                    f'answer_{turn + 1}': answer,
                }
                for conversation, question, answer, info_items, persuasion in zip(
                    running_conversations_for_caching, 
                    interviewer_questions, 
                    interviewee_answers, 
                    info_items_to_use,
                    persuasion_levels
                )
            ]

        #### 3. Handle the FINAL interviewer question and source answer outside the loop
        # Last interviewer question (ending prompt)
        # The following section handles the final question from the interviewer and the corresponding response from the source.
        # It generates prompts for the interviewer's final question using the `get_interviewer_ending_prompt` function.
        # These prompts are based on the current conversation and the outline, with a straightforward approach.
        interviewer_ending_prompts = [
            get_interviewer_ending_prompt(current_conversation, outline, "straightforward")
            for current_conversation, outline in zip(current_conversations, outlines)
        ]
        
        # The generated prompts are then passed to the `generate_INTERVIEWER_response_batch` function to obtain the final questions.
        ending_interviewer_responses = generate_INTERVIEWER_response_batch(interviewer_ending_prompts, interviewer_model)
        
        # The responses are processed to extract the text inside brackets, which is considered the final question.
        # If no text is found inside brackets, the entire response is used.
        ending_interviewer_questions = [
            extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"answer not in brackets:\n {response}"
            for response in ending_interviewer_responses
        ]
        
        # The current conversation is updated by appending the interviewer's final question to it.
        current_conversations = [
            f"{ch}\nInterviewer: {question}"
            for ch, question in zip(current_conversations, ending_interviewer_questions)
        ]
        
        # Next, prompts for the source's final response are generated using the `get_source_ending_prompt` function.
        # These prompts are based on the updated conversation, relevant information items, and the persona of the interviewee.
        ending_source_prompts = [
            get_source_ending_prompt(current_conversation, persona)
            for current_conversation, persona in zip(current_conversations, personas)
        ]
        
        # The generated prompts are then passed to the `generate_SOURCE_response_batch` function to obtain the final responses.
        ending_interviewee_responses = generate_SOURCE_response_batch(ending_source_prompts, source_model)
        
        # The responses are processed to extract the text inside brackets, which is considered the final answer.
        ending_interviewee_answers = [extract_text_inside_brackets(response) for response in ending_interviewee_responses]
        
        # The current conversation is updated by appending the interviewee's final answer to it.
        current_conversations = [
            f"{ch}\nInterviewee: {response}"
            for ch, response in zip(current_conversations, ending_interviewee_answers)
        ]
        running_conversations_for_caching = [
            {
                **conversation,
                f'question_{num_turns}': question,
                f'answer_{num_turns}': answer,
            }   
            for conversation, question, answer in zip(
                running_conversations_for_caching,
                ending_interviewer_questions,
                ending_interviewee_answers
            )
        ]
        
        # Save the batch of conducted interviews to a JSONL file
        unique_info_item_counts[start_idx: end_idx] = [len(info_set) for info_set in unique_info_items_sets]
        batch_output_df = pd.DataFrame({
            'id': batch_df['id'],
            'combined_dialogue': batch_df['combined_dialogue'],
            'info_items': batch_df['info_items'],
            'outlines': batch_df['outlines'],
            'running_conversations': running_conversations_for_caching,
            'final_conversations': current_conversations,
            'total_info_items_extracted': unique_info_item_counts[start_idx:end_idx],
            'total_info_item_count': total_info_item_counts[start_idx:end_idx],
        })

        batch_file_name = f"conducted_interviews_batch_{start_idx}_{end_idx}.jsonl"
        batch_file_path = os.path.join(output_dir, batch_file_name)
        batch_output_df.to_json(batch_file_path, orient='records', lines=True)
        print(f"Batch {start_idx} to {end_idx} saved to {batch_file_path}")

    final_df = stitch_csv_files(output_dir, f'all_{game_level}_interviews_conducted.jsonl')
    return final_df


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
        output_dir="output_results/game_sim/conducted_interviews_advanced/human_eval",
        interview_id=None
):
    os.makedirs(output_dir, exist_ok=True)
    
    role = input(f"{PROMPT_COLOR}Would you like to play as the interviewer (A) or source (B)? Please type 'A' or 'B': {RESET}").upper()
    while role not in ["A", "B"]:
        role = input(f"{ERROR_COLOR}Invalid input. Please type 'A' for interviewer or 'B' for source: {RESET}").upper()
    role = "interviewer" if role == "A" else "source"
    print(f"\n{PROMPT_COLOR}You've chosen to play as the {role}.{RESET}\n")

    persona_types = [
        "anxious",
        "avoidant",
        "adversarial",
        "defensive", 
        "straightforward", 
        "poor explainer", 
        "dominating", 
        "clueless"
    ]
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
    
    if interview_id is not None:
        sample = available_df[available_df['id'] == interview_id].iloc[0].copy()
    else:
        sample = available_df.sample(n=1).iloc[0].copy()
    sample['info_items_dict'] = ast.literal_eval(sample['info_items_dict'])
    info_items = sample['info_items']
    outline = sample['outlines']

    current_conversation = ""
    unique_info_items_set = set()
    total_info_item_count = count_information_items(info_items)
    persuasion_levels = [] ## store persuasion levels for each turn
    model = load_model(model_name)


    starting_interviewer_prompt = get_interviewer_starting_prompt(outline, "straightforward")
    interviewer_response = generate_INTERVIEWER_response_batch([starting_interviewer_prompt], model)
    interviewer_question = extract_text_inside_brackets(interviewer_response[0]) or interviewer_response[0]
    current_conversation = f"Interviewer: {interviewer_question}"

    starting_source_prompt = get_source_starting_prompt(current_conversation, persona)
    source_response = generate_SOURCE_response_batch([starting_source_prompt], model)
    source_answer = extract_text_inside_brackets(source_response[0]) or source_response[0]
    current_conversation += f"\nInterviewee: {source_answer}"
    
    #### 1. FIRST interviewer question and source answer
    if role == "interviewer":  # human is interviewer
        interviewer_prompt = f'''
        {PROMPT_COLOR}It's the beginning of the interview.

        Here is the outline of objectives you've prepared before the interview:

        {outline}

        Use this outline to help you extract as much information from the source as possible, and think about relevant follow-up questions.{RESET}
        '''
        print(f"\n{interviewer_prompt}")
        print(f"\n{INTERVIEWER_COLOR}Interviewer Starting Remark (We kicked it off for you, you'll respond next): {interviewer_question}{RESET}\n")
        print(f"\n{SOURCE_COLOR}Interviewee (LLM): {source_answer}{RESET}\n")


    else:  # human is source
        print(f"\n\n{PROMPT_COLOR}Here are all the information items you have in this interview. These represent the relevant information at your disposal to divulge to the interviewer: \n\n{info_items}{RESET}")
        print(f"\n{INTERVIEWER_COLOR}Interviewer Starting Remark (LLM): {interviewer_question}{RESET}\n")
        print(f"\n{SOURCE_COLOR}Interviewee (We kicked it off for you, you'll respond next): {source_answer}{RESET}\n")


    #### 2. Middle turns
    for turn in range(num_turns - 1):
        num_turns_left = num_turns - turn

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

            specific_info_items_prompt = get_source_specific_info_items_prompt(info_items, human_question)
            specific_info_items_response = generate_SOURCE_response_batch([specific_info_items_prompt], model)
            all_relevant_info_items = extract_text_inside_brackets(specific_info_items_response[0]) or "No information items align with the question"

            persuasion_level_prompt = get_source_persuasion_level_prompt(current_conversation, persona, persuasion_levels)
            persuasion_level_response = generate_SOURCE_response_batch([persuasion_level_prompt], model)
            persuasion_level = extract_text_inside_brackets(persuasion_level_response[0]) or "0"

            info_item_numbers = extract_information_item_numbers(all_relevant_info_items)
            persuasion_level_int = int(persuasion_level) if persuasion_level.isdigit() else 0
            persuasion_levels.append(persuasion_level_int)
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

            specific_info_item_prompt = get_source_specific_info_items_prompt(info_items, interviewer_question)
            # print(specific_info_item_prompt)
            interviewee_specific_item_responses = generate_SOURCE_response_batch([specific_info_item_prompt], model)
            all_relevant_info_items = extract_text_inside_brackets(interviewee_specific_item_responses[0]) if extract_information_item_numbers(extract_text_inside_brackets(interviewee_specific_item_responses[0])) else "No information items align with the question"
            info_item_numbers = extract_information_item_numbers(all_relevant_info_items)  # list of numbers
            
            human_persuation_criteria = f'''
            {PROMPT_COLOR}- 1: The conversation to this point is not persuasive at all and does nothing to help you trust them more.
            - 2: The conversation to this point is mildly persuasive and the journalist said a few words, once, that made you feel a little more comfortable.. You are a little willing to engage.
            - 3: The conversation to this point is persuasive enough and the journalist has repeated phrases that have made you comfortable. You are becoming willing to engage and trust them.
            - 4: The conversation to this point is very persuasive. The journalist has acknowledged your feelings, your personal identity, and your specific concerns in ways you resonate with. You are willing to engage and trust them.
            - 5: You feel totally comfortable and opened up at this stage. The journalist has acknowledged your feelings and your personal identity, very specific concerns, has connected with you in ways you resonate with. You are totally willing to engage and trust them.{RESET}
            '''
            print(f"\nPlease analyze the interviewer's last question. Given that you are a {persona} source, do you feel persuaded?\n\nEvaluate on the following criteria: \n\n{human_persuation_criteria}\n\n")
            human_persuasion_level = input(f"\n{PROMPT_COLOR}Now, please respond with 1-5: {RESET}")
            human_persuasion_level_int = int(human_persuasion_level) if human_persuasion_level.isdigit() else 0

            llm_persuasion_level_prompt = get_source_persuasion_level_prompt(
                current_conversation, 
                persona, 
                list(map(lambda x: x[1][1], persuasion_levels))
            )
            llm_persuasion_level_response = generate_SOURCE_response_batch([llm_persuasion_level_prompt], model)
            llm_persuasion_level = extract_text_inside_brackets(llm_persuasion_level_response[0]) or "0"
            llm_persuasion_level_int = int(llm_persuasion_level) if llm_persuasion_level.isdigit() else 0
            persuasion_levels.append([('human-assessed', human_persuasion_level_int), ('llm-assessed', llm_persuasion_level_int)])
            
            sampled_relevant_info_items_str, sampled_item_numbers = get_relevant_info_items(
                info_item_numbers, 
                sample['info_items_dict'], 
                persona, 
                human_persuasion_level_int,
                unique_info_items_set
            )
            unique_info_items_set.update(sampled_item_numbers)

            human_source_prompt = f'''
            {PROMPT_COLOR}Here is the list of relevant information items:
                
            {get_all_relevant_info_items(info_item_numbers, sample['info_items_dict'])}

            >> From this list, and based on your persuasion level, we suggest you share the following information items (but you can choose others if you'd like):

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
        'persuasion_levels': [persuasion_levels]
    })

    if role == "source":
        output_path = os.path.join(output_dir, f"interview_{sample['id']}_human_{persona}_{role}_vs_LLM_interviewer.jsonl")
    elif role == "interviewer":
        output_path = os.path.join(output_dir, f"interview_{sample['id']}_human_{role}_vs_LLM_{persona}_source.jsonl")
    output_df.to_json(output_path, orient='records', lines=True)
    print(f"{PROMPT_COLOR}Interview saved to {output_path}{RESET}")

    return output_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_turns", type=int, default=4, help="Number of turns in the interview")
    parser.add_argument("--model_name", type=str, default="meta-llama/meta-llama-3.1-70b-instruct", help="Model name")
    parser.add_argument("--interviewer_model_name", type=str, default="meta-llama/meta-llama-3.1-70b-instruct", help="Interviewer model name")
    parser.add_argument("--source_model_name", type=str, default="gpt-4o", help="Source model name")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for conducting interviews")
    parser.add_argument("--dataset_path", type=str, default="output_results/game_sim/outlines/final_df_with_outlines.csv", help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="output_results/game_sim/conducted_interviews_advanced", help="Output directory for saving conducted interviews")
    parser.add_argument("--human_eval", action="store_true", help="Conduct human evaluation")
    parser.add_argument("--game_level", type=str, default="advanced", help="Game level for conducting interviews")
    parser.add_argument("-id", "--interview_id", type=str, default=None, help="Interview ID for human evaluation. If not given, chosen at random")
    args = parser.parse_args()

    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_path, 'news-interview-question-generation')
    dataset_path = os.path.join(project_root, args.dataset_path)
    df = pd.read_csv(dataset_path)
    df = df.head(args.batch_size)

    if args.human_eval:
        human_evaluation = human_eval(args.num_turns, df, model_name=args.model_name, output_dir=args.output_dir, interview_id=args.interview_id)
        print(human_evaluation)
    else:
        conducted_interviews = conduct_advanced_interviews_batch(
            args.num_turns, df, 
            interviewer_model_name=args.interviewer_model_name, 
            source_model_name=args.source_model_name, 
            batch_size=args.batch_size, 
            output_dir=args.output_dir,
            game_level=args.game_level
        )
        print(conducted_interviews)

"""
python conduct_interviews_advanced.py \
    --interviewer_model_name "gpt-4o-mini" \
    --source_model_name "gpt-4o-mini" \
    --batch_size 5 \
    --dataset_path "output_results/game_sim/outlines/final_df_with_outlines.csv" \
    --output_dir "test" 
"""        

"""
python conduct_interviews_advanced.py \                                       
    --model_name "gpt-4o" \
    --batch_size 5 \
    --dataset_path "output_results/game_sim/outlines/final_df_with_outlines.csv" \
    --output_dir "test" --human_eval
"""

    # simulated_interviews = conduct_advanced_interviews_batch(num_turns, df, model_name="meta-llama/meta-llama-3.1-70b-instruct")
    # print(f"dataset with simulated interviews: {simulated_interviews}\n")
    # for i, interview in enumerate(simulated_interviews['final_conversations']):
    #     print(f"Interview {i+1}:\n {interview}\n\n\n")

'''
from the dataset of interviews, from each row (interview), plug info_items into source LLM and outlines into interviewer LLM. Then, simulate interview.
column structure of the database outputted:
'id' | 'combined_dialogue' | 'info_items' | 'outlines' | 'final_conversations'
'''
