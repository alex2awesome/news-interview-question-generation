# conduct_interviews_intermediate.py

import os
import sys
import re
import pandas as pd
import random
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import (
    load_vllm_model,
    initialize_tokenizer, 
    extract_text_inside_brackets, 
    stitch_csv_files,
    generate_SOURCE_response_batch,
    generate_INTERVIEWER_response_batch,
    count_information_items,
    extract_information_item_numbers
)
from game_sim.game_sim_prompts import (
    get_source_prompt_intermediate,
    get_source_starting_prompt,
    get_source_ending_prompt,
    get_source_specific_info_items_prompt,
    get_interviewer_prompt,
    get_interviewer_starting_prompt,
    get_interviewer_ending_prompt
)
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


# select a random number of information items, ensures no repeats
def select_info_items(info_item_numbers, used_info_items_set):
    N = len(info_item_numbers)
    if N == 0:
        return [], used_info_items_set
    k = random.randint(1, N)
    available_items = list(set(info_item_numbers) - used_info_items_set)
    if not available_items:
        return [], used_info_items_set
    num_items_to_select = min(k, len(available_items))
    selected_numbers = random.sample(available_items, num_items_to_select)
    used_info_items_set.update(selected_numbers)
    return selected_numbers, used_info_items_set

def conduct_intermediate_interviews_batch(num_turns, df, model_name="meta-llama/meta-llama-3.1-70b-instruct", batch_size=50, output_dir="output_results/game_sim/conducted_interviews_intermediate"):
    os.makedirs(output_dir, exist_ok=True)
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    df['info_items_dict'] = df['info_items_dict'].apply(json.loads)

    num_samples = len(df)
    unique_info_item_counts = [0] * num_samples
    total_info_item_counts = [0] * num_samples

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
        used_info_items_sets = [set() for _ in range(end_idx - start_idx)]
        total_info_item_counts[start_idx:end_idx] = [count_information_items(info_item) for info_item in info_items_list]
        
        personas = [random.choice(persona_types) for _ in range(end_idx - start_idx)]

        all_info_item_numbers_list = [
            extract_information_item_numbers(info_items)
            for info_items in info_items_list
        ]

        #### 1. Handle the FIRST interviewer question and source answer outside the loop
        # First interviewer question (starting prompt)
        starting_interviewer_prompts = [
            get_interviewer_starting_prompt(outline, "straightforward")
            for outline in outlines
        ]
        starting_interviewer_responses = generate_INTERVIEWER_response_batch(starting_interviewer_prompts, model, tokenizer)
        starting_interviewer_questions = [
            extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"Answer not in brackets:\n{response}"
            for response in starting_interviewer_responses
        ]
        current_conversations = [
            f"{ch}\nInterviewer: {question}"
            for ch, question in zip(current_conversations, starting_interviewer_questions)
        ]

        starting_source_prompts = [
            get_source_starting_prompt(current_conversation, info_items, persona)
            for current_conversation, info_items, persona in zip(current_conversations, info_items_list, personas)
        ]
        starting_interviewee_responses = generate_SOURCE_response_batch(starting_source_prompts, model, tokenizer)
        starting_interviewee_answers = [extract_text_inside_brackets(response) for response in starting_interviewee_responses]
        current_conversations = [
            f"{ch}\nInterviewee: {response}"
            for ch, response in zip(current_conversations, starting_interviewee_answers)
        ]

        #### 2. Handle the middle questions/answers within the loop
        for turn in range(num_turns - 2):
            num_turns_left = num_turns - (2 + turn)
            interviewer_prompts = [
                get_interviewer_prompt(current_conversation, outline, num_turns_left, "straightforward")
                for current_conversation, outline in zip(current_conversations, outlines)
            ]
            interviewer_responses = generate_INTERVIEWER_response_batch(interviewer_prompts, model, tokenizer)
            interviewer_questions = [
                extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"Answer not in brackets:\n{response}"
                for response in interviewer_responses
            ]
            gc.collect()

            current_conversations = [
                f"{ch}\nInterviewer: {question}"
                for ch, question in zip(current_conversations, interviewer_questions)
            ]

            specific_info_item_prompts = [
                get_source_specific_info_items_prompt(current_conversation, info_items)
                for current_conversation, info_items in zip(current_conversations, info_items_list)
            ]
            interviewee_all_relevant_info_items_responses = generate_SOURCE_response_batch(specific_info_item_prompts, model, tokenizer)

            all_relevant_info_items = [
                extract_text_inside_brackets(response) if extract_information_item_numbers(extract_text_inside_brackets(response)) else "No information items align with the question."
                for response in interviewee_all_relevant_info_items_responses
            ]

            selected_info_items_content_list = []
            for idx, relevant_info_items_str in enumerate(all_relevant_info_items):
                relevant_info_item_numbers = extract_information_item_numbers(relevant_info_items_str)
                used_info_items_set = used_info_items_sets[idx]
                info_items_dict_sample = info_items_dict.iloc[idx]

                if relevant_info_item_numbers:
                    unique_info_items_sets[idx].update(relevant_info_item_numbers)
                    selected_numbers, updated_used_set = select_info_items(
                        info_item_numbers=relevant_info_item_numbers,
                        used_info_items_set=used_info_items_set
                    )
                    used_info_items_sets[idx] = updated_used_set

                    if selected_numbers:
                        selected_info_items_content = []
                        for num in selected_numbers:
                            key = f"Information item #{num}"
                            content = info_items_dict_sample.get(key, "")
                            if content:
                                selected_info_items_content.append(f"{key}: {content}")
                            else:
                                selected_info_items_content.append(f"{key}: [Content not found]")
                        selected_info_item_str = '\n'.join(selected_info_items_content)
                        selected_info_items_content_list.append(selected_info_item_str)
                    else:
                        selected_info_items_content_list.append("There are no information items left")
                else:
                    selected_info_items_content_list.append("No information items align with the question")

            source_prompts = [
                get_source_prompt_intermediate(current_conversation, selected_info_item, persona)
                for current_conversation, selected_info_item, persona in zip(current_conversations, selected_info_items_content_list, personas)
            ]
            interviewee_responses = generate_SOURCE_response_batch(source_prompts, model, tokenizer)
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
        ending_interviewer_responses = generate_INTERVIEWER_response_batch(interviewer_ending_prompts, model, tokenizer)
        ending_interviewer_questions = [
            extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"answer not in brackets:\n {response}"
            for response in ending_interviewer_responses
        ]
        current_conversations = [
            f"{ch}\nInterviewer: {question}"
            for ch, question in zip(current_conversations, ending_interviewer_questions)
        ]

        ending_source_prompts = [
            get_source_ending_prompt(current_conversation, persona)
            for current_conversation, persona in zip(current_conversations, personas)
        ]
        ending_interviewee_responses = generate_SOURCE_response_batch(ending_source_prompts, model, tokenizer)
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
            'total_info_item_count': total_info_item_counts[start_idx:end_idx]
        })

        batch_file_name = f"conducted_interviews_batch_{start_idx}_{end_idx}.csv"
        batch_file_path = os.path.join(output_dir, batch_file_name)
        batch_output_df.to_csv(batch_file_path, index=False)
        print(f"Batch {start_idx} to {end_idx} saved to {batch_file_path}")

    final_df = stitch_csv_files(output_dir, 'all_intermediate_interviews_conducted.csv')
    return final_df

if __name__ == "__main__":
    data_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/game_sim/outlines/final_df_with_outlines.csv"
    df = pd.read_csv(data_path)
    print(df)

    num_turns = 8
    simulated_interviews = conduct_intermediate_interviews_batch(num_turns, df, model_name="meta-llama/meta-llama-3.1-70b-instruct")
    print(simulated_interviews)
    
    # print(f"dataset with simulated interviews: {simulated_interviews}\n")
    # for i, interview in enumerate(simulated_interviews['final_conversations']):
    #     print(f"Interview {i+1}:\n {interview}\n\n\n")

'''
from the dataset of interviews, from each row (interview), plug info_items into source LLM and outlines into interviewer LLM. Then, simulate interview.
column structure of the database outputted:
'id' | 'combined_dialogue' | 'info_items' | 'outlines' | 'final_conversations'
'''
