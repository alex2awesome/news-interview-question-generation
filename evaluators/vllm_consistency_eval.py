# vllm-consistency-eval.py

import sys
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from transformers import AutoTokenizer
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import vllm_infer, vllm_infer_batch, load_vllm_model, extract_text_inside_brackets
from prompts import get_consistency_eval_prompt

def consistency_eval_prompt_loader(transcript_context, llm_question, human_question, LLM_question_type, Actual_question_type):
    prompt = get_consistency_eval_prompt(transcript_context, llm_question, human_question, LLM_question_type, Actual_question_type)
    messages = [
        {"role": "system", "content": "You are an expert at gauging similarity between any two questions."},
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
    outputs = vllm_infer_batch(messages_batch, model)
    similarity_scores = [extract_text_inside_brackets(output) for output in outputs]
    similarity_scores = [
                            1 if result.lower() in ["similar", "high"]
                            else 0 if result.lower() in ["different", "not similar", "low"]
                            else f"Error: {result}"
                            for result in similarity_scores
                        ]
    return similarity_scores

# adds similarity column to the df
def consistency_compare_process_dataset(df, output_dir="output_results", batch_size=50, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):  
    classified_similarity_results = []

    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for start_idx in range(0, len(df), batch_size):
        batch = df.iloc[start_idx:start_idx + batch_size]

        QA_Sequences = batch['QA_Sequence']
        LLM_questions = batch['LLM_Question']
        Actual_questions = batch['Actual_Question']
        LLM_question_types = batch['LLM_Question_Type']
        Actual_question_types = batch['Actual_Question_Type']

        classified_similarities = consistency_compare_batch(QA_Sequences, LLM_questions, Actual_questions, LLM_question_types, Actual_question_types, model, tokenizer)
        classified_similarity_results.extend(classified_similarities)

        gc.collect()
    
    df['Classified_Similarity'] = classified_similarity_results

    output_file_path = os.path.join(output_dir, 'LLM_consistency_eval_results.csv')
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file_path, index=False)
    print(f"csv file saved to {output_file_path}")
    return df

# ----- For Testing ----- #

# for testing (batching)
def for_testing_consistency_compare_batch(transcript_contexts, llm_questions, human_questions, LLM_question_types, Actual_question_types, model, tokenizer):
    messages_batch = [consistency_eval_prompt_loader(context, llm_question, human_question, llm_q_type, human_q_type) for context, llm_question, human_question, llm_q_type, human_q_type in zip(transcript_contexts, llm_questions, human_questions, LLM_question_types, Actual_question_types)]
    outputs = vllm_infer_batch(messages_batch, model)
    similarity_scores = [extract_text_inside_brackets(output) for output in outputs]
    similarity_scores = [
                            1 if result.lower() in ["similar", "high"]
                            else 0 if result.lower() in ["different", "not similar", "low"]
                            else f"Error: {result}"
                            for result in similarity_scores
                        ]
    return similarity_scores, outputs

# adds similarity column to the df
def for_testing_consistency_compare_process_dataset(df, output_dir="output_results", batch_size=50, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):  
    classified_similarity_results = []
    LLM_motivation = []

    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for start_idx in range(0, len(df), batch_size):
        batch = df.iloc[start_idx:start_idx + batch_size]

        QA_Sequences = batch['QA_Sequence']
        LLM_questions = batch['LLM_Question']
        Actual_questions = batch['Actual_Question']
        LLM_question_types = batch['LLM_Question_Type']
        Actual_question_types = batch['Actual_Question_Type']

        classified_similarities, motivation_responses = for_testing_consistency_compare_batch(QA_Sequences, LLM_questions, Actual_questions, LLM_question_types, Actual_question_types, model, tokenizer)
        classified_similarity_results.extend(classified_similarities)
        LLM_motivation.extend(motivation_responses)

        gc.collect()
    
    df['Classified_Similarity'] = classified_similarity_results
    df['LLM Reasoning (Motivation for Classification)'] = LLM_motivation

    output_file_path = os.path.join(output_dir, 'LLM_consistency_eval_results.csv')
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file_path, index=False)
    print(f"csv file saved to {output_file_path}")
    return df

if __name__ == "__main__":
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/CoT_outline/LLM_classified_results.csv"
    df = pd.read_csv(dataset_path)
    df = df.head(50)
    print(df)

    new_df = for_testing_consistency_compare_process_dataset(df, output_dir="output_results/test/consistency_eval", model_name="meta-llama/Meta-Llama-3-8B-Instruct") # saves consistency_eval labels in LLM_consistency_eval_results.csv
    print(new_df)

    # filtered_df = df[(df["Classified_Similarity"].str.contains("Error", na=False))]
    # Classified_Similarity = filtered_df["Classified_Similarity"].tolist()
    # count = 0
    # for label in Classified_Similarity:
    #     count += 1
    #     print(label)
    # print(f"proportion of the errors in a sample of 1000 data points: {count/df.shape[0]}")
