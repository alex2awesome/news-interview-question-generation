import os
import sys
import re
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import load_vllm_model, initialize_tokenizer, extract_text_inside_brackets, stitch_csv_files
from game_sim.game_sim_prompts import get_source_prompt, get_interviewer_prompt

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

        source_prompt = get_source_prompt(conversation_history, "", "honest")
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
        ]
        for prompt in prompts
    ]
    return vllm_infer_batch(messages_batch, model, tokenizer)

def generate_vllm_SOURCE_response_batch(prompts, model, tokenizer):
    messages_batch = [
        [
            {"role": "system", "content": "You are a guest getting interviewed"}, 
            {"role": "user", "content": prompt}
        ]
        for prompt in prompts
    ]
    return vllm_infer_batch(messages_batch, model, tokenizer)

# regex to match "Information Item {integer}" and extract the integer
def extract_information_item_numbers(response):
    return [int(num) for num in re.findall(r'(?i)information item #?(\d+)', response)]

# Regular expression to match "Information item #{integer}"
def count_information_items(info_items_text):
    return len(re.findall(r'(?i)information item #?\d+', info_items_text))

def conduct_interviews_batch(num_turns, df, model_name="meta-llama/Meta-Llama-3-70B-Instruct", batch_size=3, output_dir="output_results/game_sim/conducted_interviews"):
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    num_samples = len(df)  # Number of rows in the dataframe
    unique_info_item_counts = [0] * num_samples  # Store unique information item counts per sample
    total_info_item_counts = [0] * num_samples  # Store total number of information items per sample

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)

        batch_df = df.iloc[start_idx:end_idx]
        info_items = batch_df['info_items']
        outlines = batch_df['outlines']

        current_conversations = [""] * (end_idx - start_idx)
        unique_info_items_sets = [set() for _ in range(end_idx - start_idx)]

        # Count the total number of information items for each row in the batch
        total_info_item_counts[start_idx:end_idx] = [count_information_items(info_item) for info_item in info_items]

        for turn in range(num_turns):
            # First, interviewer asks the question
            interviewer_prompts = [
                get_interviewer_prompt(current_conversation, outline, "straightforward")
                for current_conversation, outline in zip(current_conversations, outlines)
            ]
            interviewer_responses = generate_vllm_INTERVIEWER_response_batch(interviewer_prompts, model, tokenizer)
            interviewer_questions = [extract_text_inside_brackets(response) for response in interviewer_responses]
            gc.collect()

            # Update current conversations with the interviewer questions
            current_conversations = [
                f"{ch}\nInterviewer: {question}"
                for ch, question in zip(current_conversations, interviewer_questions)
            ]

            # Then, the interviewee responds
            source_prompts = [
                get_source_prompt(current_conversation, info_item, "honest")
                for current_conversation, info_item in zip(current_conversations, info_items)
            ]
            interviewee_responses = generate_vllm_SOURCE_response_batch(source_prompts, model, tokenizer)
            for idx, response in enumerate(interviewee_responses):
                info_item_numbers = extract_information_item_numbers(response)
                unique_info_items_sets[idx].update(info_item_numbers)
            gc.collect()

            # Update current conversations with the interviewee responses
            current_conversations = [
                f"{ch}\nInterviewee: {response}"
                for ch, response in zip(current_conversations, interviewee_responses)
            ]

        # After all turns, count the unique information items mentioned
        unique_info_item_counts[start_idx:end_idx] = [len(info_set) for info_set in unique_info_items_sets]

        # Create a DataFrame for the current batch
        batch_output_df = pd.DataFrame({
            'id': batch_df['id'],
            'combined_dialogue': batch_df['combined_dialogue'],
            'info_items': batch_df['info_items'],
            'outlines': batch_df['outlines'],
            'final_conversations': current_conversations,
            'total_info_items_extracted': unique_info_item_counts[start_idx:end_idx],  # unique info items extracted by interviewer
            'total_info_item_count': total_info_item_counts[start_idx:end_idx]  # total info items the source has
        })

        # Save the batch to a CSV file with a unique name
        batch_file_name = f"conducted_interviews_batch_{start_idx}_{end_idx}.csv"
        batch_file_path = os.path.join(output_dir, batch_file_name)
        batch_output_df.to_csv(batch_file_path, index=False)
        print(f"Batch {start_idx} to {end_idx} saved to {batch_file_path}")

    final_df = stitch_csv_files(output_dir, 'all_conducted_interviews.csv')
    return final_df

if __name__ == "__main__": 
    data_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/game_sim/outlines/final_df_with_outlines.csv"
    df = pd.read_csv(data_path)
    df = df.head(9)
    print(df) # df has columns info_items and outlines
    
    num_turns = 5
    simulated_interviews = conduct_interviews_batch(num_turns, df, model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    print(f"dataset with simulated interviews: {simulated_interviews}\n")
    
    # for i, interview in enumerate(simulated_interviews['final_conversations']):
    #      print(f"Interview {i+1}:\n {interview}\n\n\n")
    
    '''

    from the dataset of interviews, from each row (interview), plug info_items into source LLM and outlines into interviewer LLM. Then, simulate interview.
    
    column structure of the database outputted:

    'id' | 'combined_dialogue' | 'info_items' | 'outlines' | 'final_conversations'
    '''
    # response = "Information Item #12, information Item 324567, information iTem #696969"
    # print(extract_information_item_numbers(response))