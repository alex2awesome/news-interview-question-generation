# vllm-consistency-eval.py

import sys
import os
import pandas as pd
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import vllm_infer, vllm_infer_batch, load_vllm_model, extract_text_inside_brackets, create_combined_dialogue_df
from prompts import DIMENSION_OF_SIMILARITY_PROMPT
# from vllm-type-classification import classify_question 
# ^include classify_question(model_name, messages) as a parameter later

def consistency_eval_prompt_loader(transcript_context, llm_question, human_question, LLM_question_type, Actual_question_type):
    prompt = DIMENSION_OF_SIMILARITY_PROMPT.format(
        transcript_context=transcript_context,
        LLM_question=llm_question,
        human_question=human_question,
        LLM_question_type=LLM_question_type,
        Actual_question_type=Actual_question_type
    )
    messages = [
        {"role": "system", "content": "You are a world-class annotator for question similarity."},
        {"role": "user", "content": prompt}
    ]
    return messages

# single-use
def consistency_compare(messages, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    generated_text = vllm_infer(messages, model_name)
    print(f"generated_text: {generated_text}")

    similarity_scores_str = extract_text_inside_brackets(generated_text)
    similarity_scores_list = similarity_scores_str.split(', ')
    print(f'new sim score: {similarity_scores_list}')
    similarity_scores = [1 if score.lower() == 'yes' else 0 for score in similarity_scores_list]
    print(f'old sim score: {similarity_scores}')
    
    def similarity_score(scores): 
        return sum(scores)
    
    return similarity_score(similarity_scores)

# for batching
def consistency_compare_batch(transcript_contexts, llm_questions, human_questions, LLM_question_types, Actual_question_types, model, tokenizer):
    messages_batch = [consistency_eval_prompt_loader(context, llm_question, human_question, llm_q_type, human_q_type) for context, llm_question, human_question, llm_q_type, human_q_type in zip(transcript_contexts, llm_questions, human_questions, LLM_question_types, Actual_question_types)]
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    outputs = vllm_infer_batch(formatted_prompts, model)
    similarity_score = [extract_text_inside_brackets(output) for output in outputs]
    return similarity_score

# adds similarity column to the df
def consistency_compare_process_dataset(df, output_dir="output_results", batch_size=100, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):  
    classified_similarity_results = []

    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for start_idx in range(0, len(df), batch_size):
        batch = df.iloc[start_idx:start_idx + batch_size]

        QA_Sequences = batch['QA_Sequence'].tolist()
        LLM_questions = batch['LLM_Question'].tolist()
        Actual_questions = batch['Actual_Question'].tolist()
        LLM_question_types = batch['LLM_Question_Type'].tolist()
        Actual_question_types = batch['Actual_Question_Type'].tolist()

        classified_similarities = consistency_compare_batch(QA_Sequences, LLM_questions, Actual_questions, LLM_question_types, Actual_question_types, model, tokenizer)
        classified_similarity_results.extend(classified_similarities)
    
    df['Classify_Similarity'] = classified_similarity_results

    output_file_path = os.path.join(output_dir, 'LLM_consistency_eval_results.csv')
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file_path, index=False)
    return df

if __name__ == "__main__":
    df = pd.read_csv(os.path.join("output_results", "LLM_classified_results.csv"))
    df = consistency_compare_process_dataset(df, model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    print(df.head())



    # ex_llm_question = "What are the main causes of climate change?"
    # ex_human_question = "Can you explain why the climate is changing?"
    # ex_transcript_context = "We are discussing environmental issues, particularly focusing on climate change and its causes."
    # messages = consistency_eval_prompt_loader(ex_transcript_context, ex_llm_question, ex_human_question)
    
    # sim_score = consistency_compare(messages, "meta-llama/Meta-Llama-3-8B-Instruct")
    # print(f'Total similarity score: {sim_score}')

    # '''
    # Future implementation: 
    # 1. Incorporate function call to classify_question(model_name, messages) 
    # and include it as an input to make the LLM evalution function more robust.
    # '''