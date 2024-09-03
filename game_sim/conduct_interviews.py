import os
import sys
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import load_vllm_model, initialize_tokenizer, extract_text_inside_brackets
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

def conduct_interviews_batch(num_turns, df, model_name="meta-llama/Meta-Llama-3-70B-Instruct", batch_size=100, output_dir="output_results/game_sim/conducted_interviews"):
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    num_samples = len(df) # if 5 rows, num_samples = 5
    final_conversations = [""] * num_samples # this list is to store the final conversation
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        
        batch_df = df.iloc[start_idx:end_idx]
        info_items = batch_df['info_items']
        outlines = batch_df['outlines']
        
        current_conversations = [""] * (end_idx - start_idx)
        
        for turn in range(num_turns):
            # First, interviewer asks the question
            interviewer_prompts = [
                get_interviewer_prompt(current_conversation, outline, "straightforward")
                for current_conversation, outline in zip(current_conversations, outlines)
            ]
            interviewer_responses = generate_vllm_INTERVIEWER_response_batch(interviewer_prompts, model, tokenizer)
            interviewer_questions = [extract_text_inside_brackets(response) for response in interviewer_responses]

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

            # Update current conversations with the interviewee responses
            current_conversations = [
                f"{ch}\nInterviewee: {response}"
                for ch, response in zip(current_conversations, interviewee_responses)
            ]

        final_conversations[start_idx:end_idx] = current_conversations

    new_df = pd.DataFrame({
        'id': df['id'],
        'combined_dialogue': df['combined_dialogue'],
        'info_items': df['info_items'],
        'outlines': df['outlines'],
        'final_conversations': final_conversations
    })
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "conducted_interviews.csv")
    new_df.to_csv(output_file_path, index=False)
    print(f"CSV file saved to {output_file_path}")
    return new_df

if __name__ == "__main__": 
    data_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/game_sim/outlines/final_df_with_outlines.csv"
    df = pd.read_csv(data_path)
    df = df.head(3)
    print(df) # df has columns info_items and outlines
    
    num_turns = 5
    simulated_interviews = conduct_interviews_batch(num_turns, df, model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    print(f"dataset with simulated interviews: {simulated_interviews}\n")
    
    for i, interview in enumerate(simulated_interviews['final_conversations']):
         print(f"Interview {i+1}:\n {interview}\n\n\n")
    
    '''

    from the dataset of interviews, from each row (interview), plug info_items into source LLM and outlines into interviewer LLM. Then, simulate interview.
    
    column structure of the database outputted:

    'id' | 'combined_dialogue' | 'info_items' | 'outlines' | 'final_conversations'
    '''