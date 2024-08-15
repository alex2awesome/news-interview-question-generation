# classify_all_questions.py

import sys
import os
import re
import gc
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import load_vllm_model, vllm_infer_batch, extract_text_inside_brackets
from prompts import CLASSIFY_ALL_QUESTIONS_USING_TAXONOMY_PROMPT, TAXONOMY
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def all_questions_type_classification_prompt_loader(QA_seq, question):
    prompt = CLASSIFY_ALL_QUESTIONS_USING_TAXONOMY_PROMPT.format(transcript=QA_seq, question=question)
    messages = [
        {"role": "system", "content": "You are a world-class annotator for interview questions."},
        {"role": "user", "content": prompt}
    ]
    return messages

def classify_question_batch(QA_Sequences, questions, model, tokenizer):
    messages_batch = [all_questions_type_classification_prompt_loader(QA_seq, question) for QA_seq, question in zip(QA_Sequences, questions)]
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    outputs = vllm_infer_batch(formatted_prompts, model)
    question_types = [extract_text_inside_brackets(output) if extract_text_inside_brackets(output).lower() in TAXONOMY else "Error" for output in outputs]
    return question_types

def extract_interviewer_questions(utt, speaker):
    questions = []
    answers = []
    current_question = []
    current_answer = []

    for i in range(len(utt)):
        if 'host' in speaker[i].lower():
            if current_answer:
                questions.append(" ".join(current_question))
                answers.append(" ".join(current_answer))
                current_answer = []
                current_question = []
            current_question.append(utt[i])
        else:
            current_answer.append(utt[i])

    if current_question and current_answer:
        questions.append(" ".join(current_question))
        answers.append(" ".join(current_answer))

    min_length = min(len(questions), len(answers))
    questions = questions[:min_length]
    answers = answers[:min_length]

    return questions, answers

def classify_each_question(df, output_dir="/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/all_questions_classified", num_interviews=150, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    os.makedirs(output_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(output_dir) if re.match(r'interview_NPR-(\d+)\.csv', f)]    
    if existing_files:
        processed_interview_ids = sorted([int(re.search(r'interview_NPR-(\d+)\.csv', f).group(1)) for f in existing_files])
        last_processed_id = processed_interview_ids[-1]
        total_len = len(processed_interview_ids)
        print(f"Total number of interviews already processed out of the original {num_interviews}: {total_len} \nLast processed id: {last_processed_id}. \nResuming starting from the next interview.")
        num_interviews -= total_len
    else:
        processed_interview_ids = []
        last_processed_id = None
        print("no existing interviews found, starting from beginning")

    unique_interviews = df['id'].unique()[:num_interviews]
    df = df[df['id'].isin(unique_interviews)]

    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for interview_id in unique_interviews:
        interview_num = int(interview_id.split('-')[1])
        if last_processed_id is not None and interview_num <= last_processed_id:
            continue
        try:
            interview_df = df[df['id'] == interview_id]
            interview_df = interview_df.reset_index(drop=True)

            interview_url = interview_df['url'].iloc[0]
            utt = eval(interview_df['utt'].iloc[0])
            speaker = eval(interview_df['speaker'].iloc[0])
            combined_dialogue = interview_df['combined_dialogue'].iloc[0]

            questions, answers = extract_interviewer_questions(utt, speaker)
            question_types = classify_question_batch(combined_dialogue, questions, model, tokenizer)
            
            interview_data = {
                'Interview id': [interview_id] + [''] * (len(questions) - 1),
                'interview url': [interview_url] + [''] * (len(questions) - 1),
                'combined dialogue': [combined_dialogue] + [''] * (len(questions) - 1),
                'Question (Interviewer)': questions,
                'Answer (Guest)': answers,
                'Question type': question_types
            }
            interview_df_final = pd.DataFrame(interview_data)

            output_file_path = os.path.join(output_dir, f'interview_{interview_id}.csv')
            os.makedirs(output_dir, exist_ok=True)
            interview_df_final.to_csv(output_file_path, index=False)

            print(f"Processed and saved interview {interview_id} to {output_file_path}")

            gc.collect()

        except Exception as e:
            print(f"Error processing interview {interview_id}: {e}")
            break

    print("All interviews processed and saved.")

if __name__ == "__main__":
    df = pd.read_csv("/project/jonmay_231/spangher/Projects/news-interview-question-generation/dataset/final_dataset.csv")
    print(df)
    classify_each_question(df, num_interviews=150, model_name="meta-llama/Meta-Llama-3-70B-Instruct")
    print('done running!')