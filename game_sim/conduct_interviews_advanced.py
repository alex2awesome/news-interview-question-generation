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
from helper_functions import load_vllm_model, initialize_tokenizer, extract_text_inside_brackets, stitch_csv_files
from game_sim.game_sim_prompts import get_source_starting_prompt, get_source_ending_prompt, get_source_specific_info_item_prompt, get_source_persuasion_level_prompt, get_advanced_source_persona_prompt, get_advanced_interviewer_prompt, get_interviewer_starting_prompt, get_interviewer_ending_prompt, AVOIDANT_PROMPT, DEFENSIVE_PROMPT, EVASIVE_PROMPT, STRAIGHTFORWARD_PROMPT

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

# selects random segments from a given information item, returning the formatted segments and an updated used_segments_dict.
def get_random_segments(segmented_info_items_str, chosen_info_item, used_segments_dict, max_proportion=1.0, persuasion_level=1, persona='avoidant'):
    try:
        segmented_info_items = ast.literal_eval(segmented_info_items_str)
    except:
        return "Error: Unable to parse segmented info items.", used_segments_dict, None

    item_number = int(re.search(r'#(\d+)', chosen_info_item).group(1))
    item_key = f"Information item #{item_number}"

    if item_key not in segmented_info_items:
        return "No segments found for this information item.", used_segments_dict, None

    segments = segmented_info_items[item_key]
    total_segments = len(segments)

    if item_key not in used_segments_dict:
        used_segments_dict[item_key] = set()

    available_segments = [seg for i, seg in enumerate(segments) if i not in used_segments_dict[item_key]]

    if not available_segments:
        return "All segments for this item have been used.", used_segments_dict, None

    proportion_to_return = random.uniform(0, min(max_proportion, 1.0))
    
    num_segments_to_return = round(proportion_to_return * len(available_segments))
    num_segments_to_return = max(1, num_segments_to_return)  

    selected_segments = random.sample(available_segments, num_segments_to_return)

    for seg in selected_segments:
        used_segments_dict[item_key].add(segments.index(seg))

    formatted_segments = "\n".join(f"- {segment}" for segment in selected_segments)

    return formatted_segments, used_segments_dict

# parameters of the beta distribution 
PERSONA_DICT = {
    'anxious': [(2, 4), (4, 4), (4, 2)],
    'avoidant': [(1.5, 3), (4, 3), (7, 2)],
    'evasive': [],
    'straightforward': []
}

def sample_info_items(num_info_items, persuasion_level, persona):
    """
    Sample the # of info_items to return out of the total `num_info_items`

    Parameters:
        * `num_info_items`: total number of info items available to divulge
        * `persona`: the persona of the source (\in {'anxious', 'avoidant', ...}
        * `persuasion_level`: how persuaded the source is (\in {0, 1, 2})
    """

    a, b = PERSONA_DICT[persona][persuasion_level]
    x = np.linspace(0, 1, num_info_items)
    y_cdf = beta.cdf(x, a, b)
    ps = y_cdf[1:] - y_cdf[:-1]
    return np.random.choice(range(num_info_items), p=ps)

def conduct_intermediate_interviews_batch(num_turns, df, model_name="meta-llama/Meta-Llama-3-70B-Instruct", batch_size=50, output_dir="output_results/game_sim/conducted_interviews_advanced"):
    os.makedirs(output_dir, exist_ok=True)
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    num_samples = len(df)
    unique_info_item_counts = [0] * num_samples
    total_info_item_counts = [0] * num_samples

    persona_prompts = {
        "Avoidant": AVOIDANT_PROMPT,
        "Defensive": DEFENSIVE_PROMPT,
        "Evasive": EVASIVE_PROMPT,
        "Straightforward": STRAIGHTFORWARD_PROMPT
    }

    persona_types = ["Avoidant", "Defensive", "Evasive", "Straightforward"]

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
            get_source_starting_prompt(current_conversation, info_item_list)
            for current_conversation, info_item_list in zip(current_conversations, info_items)
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

            # ask source for specific information item
            specific_info_item_prompts = [
                get_source_specific_info_item_prompt(current_conversation, info_item_list)
                for current_conversation, info_item_list in zip(current_conversations, info_items)
            ]
            interviewee_specific_item_responses = generate_vllm_SOURCE_response_batch(specific_info_item_prompts, model, tokenizer)
            specific_info_items = [
                extract_text_inside_brackets(response) if extract_information_item_numbers(extract_text_inside_brackets(response)) else "No information items align with the question"
                for response in interviewee_specific_item_responses
            ]

            # ask source for level of persuasion
            persuasion_level_prompts = [
                get_source_persuasion_level_prompt(current_conversations)
                for current_conversation in current_conversations
            ]
            interviewee_persuasion_level_responses = generate_vllm_SOURCE_response_batch(persuasion_level_prompts, model, tokenizer)
            persuasion_levels = [
                extract_text_inside_brackets(response) if extract_information_item_numbers(extract_text_inside_brackets(response)) else "No"
                for response in interviewee_persuasion_level_responses
            ]
            
            # ask source for response, given specific info segments 
            random_segments = []
            for idx, (specific_item, segmented_items, persuasion_level, persona) in enumerate(zip(specific_info_items, batch_df['segmented_info_items'], persuasion_levels, personas)):
                info_item_numbers = extract_information_item_numbers(specific_item)
                unique_info_items_sets[idx].update(info_item_numbers)
                
                if info_item_numbers:
                    chosen_item = f"Information Item #{info_item_numbers[0]}"
                    segments, used_segments_dicts[idx] = get_random_segments(segmented_items, chosen_item, used_segments_dicts[idx], persuasion_level, persona)
                    extracted_segments_sets[idx].update([seg.strip() for seg in segments.split('\n') if seg.strip()])
                else:
                    segments = "No specific information item was chosen."
                random_segments.append(segments)

            gc.collect()

            # source response
            source_prompts = [
                get_advanced_source_persona_prompt(current_conversation, info_item_list, random_segment, persona_prompts, persona_prompts[persona])
                for current_conversation, info_item_list, random_segment in zip(current_conversations, info_items, random_segments)
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
            get_source_ending_prompt(current_conversation, info_item_list)
            for current_conversation, info_item_list in zip(current_conversations, info_items)
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

if __name__ == "__main__":
    data_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/game_sim/segmented_info_items/final_df_with_segmented_info_items.csv"
    df = pd.read_csv(data_path)
    df = df.head(10)
    print(df)
    # df has columns info_items and outlines
    num_turns = 8
    simulated_interviews = conduct_intermediate_interviews_batch(num_turns, df, model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    print(f"dataset with simulated interviews: {simulated_interviews}\n")
    for i, interview in enumerate(simulated_interviews['final_conversations']):
        print(f"Interview {i+1}:\n {interview}\n\n\n")

'''
from the dataset of interviews, from each row (interview), plug info_items into source LLM and outlines into interviewer LLM. Then, simulate interview.
column structure of the database outputted:
'id' | 'combined_dialogue' | 'info_items' | 'outlines' | 'final_conversations'
'''
