# classify_all_questions.py

import sys
import os
import re
import gc
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import load_vllm_model, vllm_infer_batch, extract_text_inside_brackets, combine_csv_files
from prompts import get_classify_all_questions_taxonomy_prompt, TAXONOMY
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def all_questions_type_classification_prompt_loader(QA_seq, question):
    prompt = get_classify_all_questions_taxonomy_prompt(QA_seq, question)
    messages = [
        {"role": "system", "content": "You are a world-class annotator for interview questions."},
        {"role": "user", "content": prompt}
    ]
    return messages

# ask alex if need multi-label functionality
def classify_question_batch(QA_Sequences, questions, model, tokenizer):
    messages_batch = [all_questions_type_classification_prompt_loader(QA_seq, question) for QA_seq, question in zip(QA_Sequences, questions)]
    outputs = vllm_infer_batch(messages_batch, model)
    question_types = [extract_text_inside_brackets(output) if extract_text_inside_brackets(output).lower() in TAXONOMY else extract_text_inside_brackets(output) for output in outputs]
    return question_types

def extract_interviewer_questions(utt, speaker):
    questions = []
    answers = []
    current_question = []
    current_answer = []

    interviewer = None
    for s in speaker:
        if 'host' in s.lower():
            interviewer = s
            break
    if not interviewer:
        interviewer = speaker[0]

    # makes sure there's more than one unique speaker in the speaker list, return "not validate interview" lists
    unique_speakers = set([s.split(",")[0].strip() for s in speaker])
    if len(unique_speakers) == 1:
        half_length = len(speaker) // 2
        return ["not validate 1-on-1 interview"] * half_length, ["not validate 1-on-1 interview"] * half_length

    for i in range(len(utt)):
        if interviewer.lower() in speaker[i].lower() or speaker[i].lower() in interviewer.lower():
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
    if len(questions) == 1 or len(answers) == 1 or len(questions) == 0 or len(answers) == 0:
        return ["bad interview"] * 2, ["bad interview"] * 2
    return questions, answers

# calculates the diversity ratio of labels in a list of question types, excluding 'starting/ending remarks'
def calculate_label_diversity(question_types):
    if len(question_types) <= 4:
        return 1
    unique_labels = set(question_types)
    if "starting/ending remarks" in unique_labels:
        unique_labels.remove("starting/ending remarks")
    diversity_ratio = len(unique_labels) / len(question_types) if question_types else 0
    return diversity_ratio

def classify_each_question(df, output_dir="/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/vllm_all_questions_classified", num_interviews=150, model_name="meta-llama/Meta-Llama-3-70B-Instruct", max_retries=3):
    os.makedirs(output_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(output_dir) if re.match(r'interview_[\w-]+\.csv', f)]    
    if existing_files:
        processed_interview_ids = [re.search(r'interview_([\w-]+)\.csv', f).group(1) for f in existing_files]
        total_len = len(processed_interview_ids)
        print(f"{num_interviews} out of the {total_len} total interviews have already processed out of the original. \nResuming starting from the next interview.")
        num_interviews -= total_len
    else:
        processed_interview_ids = []
        print("no interviews have already been processed, starting from beginning")
    df = df[~df['id'].isin(processed_interview_ids)]

    unique_interviews = df['id'].unique()[:num_interviews]
    df = df[df['id'].isin(unique_interviews)]

    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for interview_id in unique_interviews:
        try:
            interview_df = df[df['id'] == interview_id]
            interview_df = interview_df.reset_index(drop=True)

            interview_url = interview_df['url'].iloc[0]
            utt = eval(interview_df['utt'].iloc[0])
            speaker = eval(interview_df['speaker'].iloc[0])
            combined_dialogue = interview_df['combined_dialogue'].iloc[0]

            questions, answers = extract_interviewer_questions(utt, speaker)
            for attempt in range(max_retries):
                question_types = classify_question_batch(combined_dialogue, questions, model, tokenizer)
                if calculate_label_diversity(question_types) >= 0.10:
                    break
                else:
                    print(f"Low diversity detected for interview {interview_id}. Retrying classification... (Attempt {attempt + 1}/{max_retries})")
            
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

    print("All interviews processed and saved.")

if __name__ == "__main__":
    df = pd.read_csv("/project/jonmay_231/spangher/Projects/news-interview-question-generation/dataset/final_dataset.csv")
    print(df)
    output_dir="/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/test/vllm_all_questions_classified"
    classify_each_question(df, output_dir, num_interviews=3, model_name="meta-llama/Meta-Llama-3-8B-Instruct")
    directory_path = 'output_results/vllm_all_questions_classified'
    output_file = 'output_results/test/vllm_all_questions_classified/all_questions_classified.csv'
    combine_csv_files(output_dir, output_file)
    