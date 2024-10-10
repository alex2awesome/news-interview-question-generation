# conduct_interviews_intermediate_v2.py
import os
import sys
import re
import pandas as pd
import random
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import (
    load_vllm_model,
    initialize_tokenizer,
    extract_text_inside_brackets,
    stitch_csv_files,
    find_project_root,
    generate_SOURCE_response_batch,
    generate_INTERVIEWER_response_batch,
    extract_information_item_numbers,
    count_information_items
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

def conduct_intermediate_interviews_batch(num_turns, df, interviewer_strategy="straightforward", interviewer_model_name="meta-llama/meta-llama-3.1-70b-instruct", source_model_name="meta-llama/meta-llama-3.1-70b-instruct", batch_size=50, output_dir="output_results/game_sim/conducted_interviews_intermediate_v2"):
    os.makedirs(output_dir, exist_ok=True)

    df['info_items_dict'] = df['info_items_dict'].apply(json.loads)

    num_samples = len(df)
    unique_info_item_counts = [0] * num_samples
    total_info_item_counts = [count_information_items(info_item) for info_item in df['info_items']]

    current_conversations = [""] * num_samples
    unique_info_items_sets = [set() for _ in range(num_samples)]
    used_info_items_sets = [set() for _ in range(num_samples)]

    persona_types = ["avoidant", "defensive", "straightforward",
                     "poor explainer", "dominating", "clueless"]
    personas = [random.choice(persona_types) for _ in range(num_samples)]

    for turn in range(num_turns):
        #### Interviewer Phase ####
        print(f"Turn {turn + 1}: Generating interviewer questions")
        interviewer_model = load_vllm_model(interviewer_model_name)
        interviewer_tokenizer = initialize_tokenizer(interviewer_model_name)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_df = df.iloc[start_idx:end_idx]
            outlines = batch_df['outlines']

            if turn == 0:
                interviewer_prompts = [
                    get_interviewer_starting_prompt(outline, interviewer_strategy)
                    for outline in outlines
                ]
            elif turn == num_turns - 1:
                interviewer_prompts = [
                    get_interviewer_ending_prompt(current_conversation, outline, interviewer_strategy)
                    for current_conversation, outline in zip(current_conversations[start_idx:end_idx], outlines)
                ]
            else:
                interviewer_prompts = [
                    get_interviewer_prompt(current_conversation, outline, interviewer_strategy)
                    for current_conversation, outline in zip(current_conversations[start_idx:end_idx], outlines)
                ]

            interviewer_responses = generate_INTERVIEWER_response_batch(interviewer_prompts, interviewer_model, interviewer_tokenizer)

            interviewer_questions = [
                extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"Answer not in brackets:\n{response}"
                for response in interviewer_responses
            ]

            current_conversations[start_idx:end_idx] = [
                f"{ch}\nInterviewer: {question}"
                for ch, question in zip(current_conversations[start_idx:end_idx], interviewer_questions)
            ]

        del interviewer_model
        gc.collect()

        #### Source Phase ####
        print(f"Turn {turn + 1}: Generating source responses")
        source_model = load_vllm_model(source_model_name)
        source_tokenizer = initialize_tokenizer(source_model_name)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_df = df.iloc[start_idx:end_idx]
            info_items_dict_list = batch_df['info_items_dict'].tolist()
            info_items_list = batch_df['info_items']
            personas_batch = personas[start_idx:end_idx]

            if turn == 0: # Starting Turn
                starting_source_prompts = [
                    get_source_starting_prompt(current_conversation, info_items, persona) 
                    for current_conversation, info_items, persona 
                    in zip(current_conversations[start_idx:end_idx], info_items_list, personas_batch)
                ]

                source_responses = generate_SOURCE_response_batch(starting_source_prompts, source_model, source_tokenizer)
                source_answers = [extract_text_inside_brackets(response) for response in source_responses]
                current_conversations[start_idx:end_idx] = [
                    f"{ch}\nInterviewee: {response}"
                    for ch, response in zip(current_conversations[start_idx:end_idx], source_answers)
                ]

            elif turn == num_turns - 1: # Final Turn

                ending_source_prompts = [
                    get_source_ending_prompt(current_conversation, persona) for current_conversation, persona 
                    in zip(current_conversations[start_idx:end_idx], personas_batch)
                ]

                source_responses = generate_SOURCE_response_batch(ending_source_prompts, source_model, source_tokenizer)
                source_answers = [extract_text_inside_brackets(response) for response in source_responses]
                current_conversations[start_idx:end_idx] = [
                    f"{ch}\nInterviewee: {response}"
                    for ch, response in zip(current_conversations[start_idx:end_idx], source_answers)
                ]

            else: # Middle Turns
                specific_info_item_prompts = [
                    get_source_specific_info_items_prompt(current_conversation, info_items)
                    for current_conversation, info_items in zip(current_conversations[start_idx:end_idx], info_items_list)
                ]
                interviewee_specific_item_responses = generate_SOURCE_response_batch(specific_info_item_prompts, source_model, source_tokenizer)

                all_relevant_info_items = [
                    extract_text_inside_brackets(response) if extract_information_item_numbers(extract_text_inside_brackets(response)) else "No information items align with the question."
                    for response in interviewee_specific_item_responses
                ]

                selected_info_items_content_list = []
                for idx, relevant_info_items_str in enumerate(all_relevant_info_items):
                    relevant_info_item_numbers = extract_information_item_numbers(relevant_info_items_str)
                    global_idx = start_idx + idx
                    used_info_items_set = used_info_items_sets[global_idx]
                    info_items_dict_sample = info_items_dict_list[idx]

                    if relevant_info_item_numbers:
                        selected_numbers, updated_used_set = select_info_items(
                            info_item_numbers=relevant_info_item_numbers,
                            used_info_items_set=used_info_items_set
                        )
                        used_info_items_sets[global_idx] = updated_used_set

                        if selected_numbers:
                            unique_info_items_sets[global_idx].update(selected_numbers)
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
                        selected_info_items_content_list.append("No information items align with the question.")

                source_prompts = [
                    get_source_prompt_intermediate(current_conversation, selected_info_item_content, persona)
                    for current_conversation, selected_info_item_content, persona in zip(
                        current_conversations[start_idx:end_idx],
                        selected_info_items_content_list,
                        personas_batch
                    )
                ]

                source_responses = generate_SOURCE_response_batch(source_prompts, source_model, source_tokenizer)
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
            'total_info_items_extracted': unique_info_item_counts,
            'total_info_item_count': total_info_item_counts   
        })

        output_file_name = os.path.join(output_dir, 'all_intermediate_interviews_conducted_v2.csv')
        output_df.to_csv(output_file_name, index=False)
        print(f"All interviews saved to {output_file_name}")

        return output_df

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_path, 'news-interview-question-generation')
    dataset_path = os.path.join(project_root, "output_results/game_sim/outlines/final_df_with_outlines.csv")
    df = pd.read_csv(dataset_path)
    df = df.head(3)
    print(df)

    num_turns = 4
    simulated_interviews = conduct_intermediate_interviews_batch(
        num_turns,
        df,
        interviewer_model_name="meta-llama/llama-3.1-8b-instruct",
        source_model_name="meta-llama/llama-3.1-8b-instruct",
        batch_size=5,
        output_dir="output_results/game_sim/conducted_interviews_intermediate_v2"
    )
    print(simulated_interviews)
    print(simulated_interviews['final_conversations'].iloc[0])
