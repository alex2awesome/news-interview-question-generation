# conduct_interviews_basic.py
import os
import sys
import re
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import load_vllm_model, initialize_tokenizer, extract_text_inside_brackets, stitch_csv_files, find_project_root
from game_sim.game_sim_prompts import get_source_prompt_basic, get_source_starting_prompt, get_source_ending_prompt, get_source_specific_info_item_prompt, get_interviewer_prompt, get_interviewer_starting_prompt, get_interviewer_ending_prompt

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

def conduct_basic_interviews_batch(num_turns, df, interviewer_model_name = "meta-llama/Meta-Llama-3-70B-Instruct", source_model_name="meta-llama/Meta-Llama-3-70B-Instruct", batch_size=50, output_dir="output_results/game_sim/conducted_interviews_basic"):
    os.makedirs(output_dir, exist_ok=True)
    source_model = load_vllm_model(source_model_name)
    source_tokenizer = initialize_tokenizer(source_model_name)

    interviewer_model = load_vllm_model(interviewer_model_name)
    interviewer_tokenizer = initialize_tokenizer(interviewer_model_name)

    num_samples = len(df)
    unique_info_item_counts = [0] * num_samples
    total_info_item_counts = [0] * num_samples

    all_prompts = [] # TEMP LINE (DELETE LATER)
    all_responses = [] # TEMP LINE (DELETE LATER) 

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_df = df.iloc[start_idx:end_idx]
        info_items = batch_df['info_items']
        outlines = batch_df['outlines']
        current_conversations = [""] * (end_idx - start_idx)
        unique_info_items_sets = [set() for _ in range(end_idx - start_idx)]
        total_info_item_counts[start_idx:end_idx] = [count_information_items(info_item) for info_item in info_items]

        #### 1. Handle the FIRST interviewer question and source answer outside the loop
        # First interviewer question (starting prompt)
        starting_interviewer_prompts = [
            get_interviewer_starting_prompt(outline, "straightforward")
            for outline in outlines
        ]
        all_prompts.extend(starting_interviewer_prompts) # TEMP LINE (DELETE LATER)

        starting_interviewer_responses = generate_vllm_INTERVIEWER_response_batch(starting_interviewer_prompts, interviewer_model, interviewer_tokenizer)
        all_responses.extend(starting_interviewer_responses) # TEMP LINE (DELETE LATER)
        starting_interviewer_questions = [
            extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"answer not in brackets:\n {response}"
            for response in starting_interviewer_responses
        ]
        current_conversations = [
            f"{ch}\nInterviewer: {question}"
            for ch, question in zip(current_conversations, starting_interviewer_questions)
        ]

        starting_source_prompts = [
            get_source_starting_prompt(current_conversation, info_item_list)
            for current_conversation, info_item_list in zip(current_conversations, info_items)
        ]
        all_prompts.extend(starting_source_prompts) # TEMP LINE (DELETE LATER)

        starting_interviewee_responses = generate_vllm_SOURCE_response_batch(starting_source_prompts, source_model, source_tokenizer)
        all_responses.extend(starting_interviewee_responses) # TEMP LINE (DELETE LATER)
        starting_interviewee_answers = [extract_text_inside_brackets(response) for response in starting_interviewee_responses]
        current_conversations = [
            f"{ch}\nInterviewee: {response}"
            for ch, response in zip(current_conversations, starting_interviewee_answers)
        ]

        #### 2. Handle the middle questions/answers within the loop
        for turn in range(num_turns - 2):
            interviewer_prompts = [
                get_interviewer_prompt(current_conversation, outline, "straightforward")
                for current_conversation, outline in zip(current_conversations, outlines)
            ]
            all_prompts.extend(interviewer_prompts) # TEMP LINE (DELETE LATER)

            interviewer_responses = generate_vllm_INTERVIEWER_response_batch(interviewer_prompts, interviewer_model, interviewer_tokenizer)
            all_responses.extend(interviewer_responses) # TEMP LINE (DELETE LATER)
            interviewer_questions = [extract_text_inside_brackets(response) if extract_text_inside_brackets(response) else f"answer not in brackets:\n {response}" for response in interviewer_responses]
            gc.collect()

            current_conversations = [
                f"{ch}\nInterviewer: {question}"
                for ch, question in zip(current_conversations, interviewer_questions)
            ]

            specific_info_item_prompts = [
                get_source_specific_info_item_prompt(current_conversation, info_item_list)
                for current_conversation, info_item_list in zip(current_conversations, info_items)
            ]
            all_prompts.extend(specific_info_item_prompts) # TEMP LINE (DELETE LATER)
            interviewee_specific_item_responses = generate_vllm_SOURCE_response_batch(specific_info_item_prompts, source_model, source_tokenizer)
            all_responses.extend(interviewee_specific_item_responses) # TEMP LINE (DELETE LATER)
            specific_info_items = [
                extract_text_inside_brackets(response) if extract_information_item_numbers(extract_text_inside_brackets(response)) else "No information items align with the question"
                for response in interviewee_specific_item_responses
            ]

            for idx, specific_item in enumerate(specific_info_items):
                info_item_numbers = extract_information_item_numbers(specific_item)
                unique_info_items_sets[idx].update(info_item_numbers)

            gc.collect()

            source_prompts = [
                get_source_prompt_basic(current_conversation, info_item_list, specific_info_item, "honest")
                for current_conversation, info_item_list, specific_info_item in zip(current_conversations, info_items, specific_info_items)
            ]
            all_prompts.extend(source_prompts) # TEMP LINE (DELETE LATER)

            interviewee_responses = generate_vllm_SOURCE_response_batch(source_prompts, source_model, source_tokenizer)
            all_responses.extend(interviewee_responses) # TEMP LINE (DELETE LATER)
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
        all_prompts.extend(interviewer_ending_prompts) # TEMP LINE (DELETE LATER)

        ending_interviewer_responses = generate_vllm_INTERVIEWER_response_batch(interviewer_ending_prompts, interviewer_model, interviewer_tokenizer)
        all_responses.extend(ending_interviewer_responses) # TEMP LINE (DELETE LATER)
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
        all_prompts.extend(ending_source_prompts) # TEMP LINE (DELETE LATER)

        ending_interviewee_responses = generate_vllm_SOURCE_response_batch(ending_source_prompts, source_model, source_tokenizer)
        all_responses.extend(ending_interviewee_responses) # TEMP LINE (DELETE LATER)
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
            'total_info_items_extracted': unique_info_item_counts[start_idx:end_idx],  # unique info items extracted by interviewer
            'total_info_item_count': total_info_item_counts[start_idx:end_idx]  # total info items the source has
        })

        batch_file_name = f"conducted_interviews_batch_{start_idx}_{end_idx}.csv"
        batch_file_path = os.path.join(output_dir, batch_file_name)
        batch_output_df.to_csv(batch_file_path, index=False)
        print(f"Batch {start_idx} to {end_idx} saved to {batch_file_path}")

    with open(os.path.join(output_dir, "prompts.txt"), "w") as prompt_file: # TEMP LINE (DELETE LATER)
        for prompt in all_prompts: # TEMP LINE (DELETE LATER)
            prompt_file.write(prompt + "\n") # TEMP LINE (DELETE LATER)

    with open(os.path.join(output_dir, "responses.txt"), "w") as response_file: # TEMP LINE (DELETE LATER)
        for response in all_responses: # TEMP LINE (DELETE LATER)
            response_file.write(response + "\n") # TEMP LINE (DELETE LATER)
    
    final_df = stitch_csv_files(output_dir, 'all_basic_interviews_conducted.csv')
    return final_df

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_path, 'news-interview-question-generation')
    dataset_path = os.path.join(project_root, "output_results/game_sim/outlines/final_df_with_outlines.csv")
    df = pd.read_csv(dataset_path)
    print(df)

    num_turns = 8
    simulated_interviews = conduct_basic_interviews_batch(num_turns, 
                                                          df, 
                                                          interviewer_model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
                                                          source_model_name="meta-llama/Meta-Llama-3-70B-Instruct",
                                                          output_dir="output_results/game_sim/conducted_interviews_basic")
    print(simulated_interviews)
    
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