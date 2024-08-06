import os
import pandas as pd
import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from helper_functions import load_vllm_model, initialize_tokenizer

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

def conduct_interview(initial_prompt, num_turns, interviewer_role, interviewee_role, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    conversation_history = initial_prompt
    print(conversation_history)

    for _ in range(num_turns):
        interviewee_response = generate_vllm_response(conversation_history + "\nInterviewee:", interviewee_role, model, tokenizer)
        print("\nInterviewee: " + interviewee_response)
        
        conversation_history += "\nInterviewee: " + interviewee_response
        
        interviewer_question = generate_vllm_response(conversation_history + "\nInterviewer:", interviewer_role, model, tokenizer)
        print("\nInterviewer: " + interviewer_question)
        
        conversation_history += "\nInterviewer: " + interviewer_question
    
    # Generate final interviewee response
    final_interviewee_response = generate_vllm_response(conversation_history + "\nInterviewee:", interviewee_role, model, tokenizer)
    conversation_history += "\nInterviewee: " + final_interviewee_response
    
    print("\nFinal Conversation:\n" + conversation_history)

# ---- batch use ---- #
def vllm_infer_batch(messages_batch, model, tokenizer):
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    outputs = model.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

def generate_vllm_response_batch(prompts, roles, model, tokenizer):
    messages_batch = [
        [{"role": "system", "content": f"{role}."}, {"role": "user", "content": prompt}]
        for prompt, role in zip(prompts, roles)
    ]
    return vllm_infer_batch(messages_batch, model, tokenizer)

def conduct_interview_batch(num_turns, df, model_name="meta-llama/Meta-Llama-3-70B-Instruct", batch_size=100, output_dir="output_results/game_sim"):
    model = load_vllm_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    num_samples = len(df)
    final_conversations = [""] * num_samples
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        
        batch_df = df.iloc[start_idx:end_idx]
        batch_prompts = batch_df['initial_prompt'].tolist()
        batch_interviewer_roles = batch_df['interviewer_role'].tolist()
        batch_interviewee_roles = batch_df['interviewee_role'].tolist()
        
        for _ in range(num_turns):
            # Generate interviewee responses in batch
            interviewee_responses = generate_vllm_response_batch(
                [ch + "\nInterviewee:" for ch in batch_prompts], 
                batch_interviewee_roles, 
                model, 
                tokenizer
            )
            
            # Update conversation histories with interviewee responses
            batch_prompts = [
                ch + "\nInterviewee: " + response 
                for ch, response in zip(batch_prompts, interviewee_responses)
            ]
            
            # Generate interviewer questions in batch
            interviewer_questions = generate_vllm_response_batch(
                [ch + "\nInterviewer:" for ch in batch_prompts], 
                batch_interviewer_roles, 
                model, 
                tokenizer
            )
            
            # Update conversation histories with interviewer questions
            batch_prompts = [
                ch + "\nInterviewer: " + question 
                for ch, question in zip(batch_prompts, interviewer_questions)
            ]
        
        # Generate final interviewee responses to ensure the interview ends with interviewee's response
        final_interviewee_responses = generate_vllm_response_batch(
            [ch + "\nInterviewee:" for ch in batch_prompts], 
            batch_interviewee_roles, 
            model, 
            tokenizer
        )
        
        # Finalize the conversation histories with the last interviewee response
        batch_final_conversations = [
            ch + "\nInterviewee: " + response 
            for ch, response in zip(batch_prompts, final_interviewee_responses)
        ]
        
        final_conversations[start_idx:end_idx] = batch_final_conversations

    df['simulated_interview'] = final_conversations
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "conducted_interviews.csv")
    df.to_csv(output_file_path, index=False)
    print(f"CSV file saved to {output_file_path}")
    return df

if __name__ == "__main__": 
    data = {
    'initial_prompt': [
        "Interviewer: Can you tell us about the most exciting recent advancements in AI?",
        "Interviewer: What are the key challenges facing AI research today?"
    ],
    'interviewer_role': [
        "You are a journalist asking interview questions", 
        "You are a senior journalist conducting a deep dive"
    ],
    'interviewee_role': [
        "You are an AI researcher answering interview questions. Please respond with full sentence dialogue. No bullet points or lists allowed", 
        "You are an AI expert providing detailed insights. Please respond with full sentence dialogue. No bullet points or lists allowed"
    ]
    }
    df = pd.DataFrame(data)
    print(df)
    num_turns = 3

    simulated_interviews = conduct_interview_batch(num_turns, df, model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    print(f"dataset with simulated interviews: {simulated_interviews}\n")
    
    for i, interview in enumerate(simulated_interviews['simulated_interview']):
        print(f"Interview{i+1}:\n {interview}")

    '''
    how batching will work:

    from the dataset of interviews, take each interview, process it into key information, store these key pieces of information in a column called 
    
    
    finally column structure of the dataset:

    id | interview transcript (QA Sequence) | information_items | initial_prompt | interviewer_role | interviewee_role | 
    '''