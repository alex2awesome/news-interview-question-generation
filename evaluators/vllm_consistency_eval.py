# vllm-consistency-eval.py

import sys
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from transformers import AutoTokenizer
import gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import (
    vllm_infer,
    infer_batch, 
    load_model, 
    extract_text_inside_brackets,
    parse_python_dict
)
from prompts import (get_consistency_eval_prompt, get_consistency_eval_prompt_multidimensional)

def consistency_eval_prompt_loader(
        transcript_context, 
        llm_question, 
        human_question, 
        LLM_question_type, 
        Actual_question_type,
        eval_type="basic"
    ):
    if eval_type == "basic":
        prompt = get_consistency_eval_prompt(transcript_context, llm_question, human_question, LLM_question_type, Actual_question_type)
    else:
        prompt = get_consistency_eval_prompt_multidimensional(transcript_context, llm_question, human_question, LLM_question_type, Actual_question_type)
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
    outputs = infer_batch(messages_batch, model)
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
    model = load_model(model_name)
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
def for_testing_consistency_compare_batch(
        transcript_contexts, 
        llm_questions, 
        human_questions, 
        LLM_question_types, 
        Actual_question_types, 
        model, 
        eval_type="basic",
        verbose=True
    ):
    messages_batch = [
        consistency_eval_prompt_loader(context, llm_question, human_question, llm_q_type, human_q_type, eval_type=eval_type) 
        for context, llm_question, human_question, llm_q_type, human_q_type 
        in zip(transcript_contexts, llm_questions, human_questions, LLM_question_types, Actual_question_types)
    ]
    outputs = infer_batch(messages_batch, model, verbose=verbose)
    if eval_type == "basic":
        similarity_scores = [extract_text_inside_brackets(output) for output in outputs]
        similarity_scores = [
                                1 if result.lower() in ["similar", "high"]
                                else 0 if result.lower() in ["different", "not similar", "low"]
                                else f"Error: {result}"
                                for result in similarity_scores
                            ]
        return similarity_scores, outputs
    else:
        parsed_dicts = [parse_python_dict(output) for output in outputs]
        similarity_scores = []
        for parsed_dict in parsed_dicts:
            output_dict = {}
            for key, value in parsed_dict.items():
                if "`yes`" in value.strip().lower():
                    output_dict[key] = 1
                elif "`no`" in value.strip().lower():
                    output_dict[key] = 0
            similarity_scores.append(output_dict)
        return similarity_scores, parsed_dicts

# adds similarity column to the df
def for_testing_consistency_compare_process_dataset(
        df, 
        output_dir="output_results", 
        batch_size=50, 
        model_name="meta-llama/Meta-Llama-3-70B-Instruct",
        eval_type="basic",
        verbose=True
):  
    classified_similarity_results = []
    LLM_motivation = []

    model = load_model(model_name)

    for start_idx in range(0, len(df), batch_size):
        batch = df.iloc[start_idx:start_idx + batch_size]

        QA_Sequences = batch['QA_Sequence']
        LLM_questions = batch['LLM_Question']
        Actual_questions = batch['Actual_Question']
        LLM_question_types = batch['LLM_Question_Type']
        Actual_question_types = batch['Actual_Question_Type']

        classified_similarities, motivation_responses = for_testing_consistency_compare_batch(
            QA_Sequences, 
            LLM_questions, 
            Actual_questions, 
            LLM_question_types, 
            Actual_question_types, 
            model, 
            eval_type=eval_type,
            verbose=verbose
        )
        classified_similarity_results.extend(classified_similarities)
        LLM_motivation.extend(motivation_responses)

        gc.collect()
    
    df['Classified_Similarity'] = classified_similarity_results
    df['LLM Reasoning (Motivation for Classification)'] = LLM_motivation

    if eval_type == "basic":
        output_file_path = os.path.join(output_dir, 'LLM_consistency_eval_results.csv')
    else:
        output_file_path = os.path.join(output_dir, 'LLM_consistency_eval_results_multidimensional.jsonl')
    os.makedirs(output_dir, exist_ok=True)
    if output_file_path.endswith(".csv"):
        df.to_csv(output_file_path, index=False)
        print(f"csv file saved to {output_file_path}")
    elif output_file_path.endswith(".jsonl"):
        df.to_json(output_file_path, orient='records', lines=True)
        print(f"jsonl file saved to {output_file_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset that needs to be processed")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for processing the dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="Model name for the LLM")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--eval_type", type=str, default="basic", help="Type of evaluation to be performed")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    if args.dataset_path is None:
        args.dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/CoT_outline/LLM_classified_results.csv"
    if args.output_dir is None:
        args.output_dir = "output_results/test/consistency_eval"

    df = pd.read_csv(args.dataset_path)
    df = df.loc[lambda df: df['LLM_Question'] != 'Guessed Question']
    if args.debug:
        df = df.head(args.batch_size)
    print(df)

    new_df = for_testing_consistency_compare_process_dataset(
        df, 
        output_dir=args.output_dir,
        model_name=args.model_name,
        eval_type=args.eval_type,
        verbose=args.verbose
    ) # saves consistency_eval labels in LLM_consistency_eval_results.csv
    print(new_df)

    # filtered_df = df[(df["Classified_Similarity"].str.contains("Error", na=False))]
    # Classified_Similarity = filtered_df["Classified_Similarity"].tolist()
    # count = 0
    # for label in Classified_Similarity:
    #     count += 1
    #     print(label)
    # print(f"proportion of the errors in a sample of 1000 data points: {count/df.shape[0]}")
