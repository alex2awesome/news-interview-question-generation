# gpt_classify_all_questions.py

import pandas as pd
import json
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompts import get_classify_all_questions_taxonomy_prompt, TAXONOMY
from helper_functions import get_openai_client, extract_text_inside_brackets
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

client = get_openai_client()

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

    return questions, answers

def generate_batched_schema_classification_prompts(df, output_dir="output_results/gpt_batching/batch_prompts", sample_size=150):
    os.makedirs(output_dir, exist_ok=True)
    random_sample_df = df.sample(n=sample_size, random_state=42)

    for idx, row in random_sample_df.iterrows():
        interview_id = row['id']
        interview_df = df[df['id'] == interview_id].reset_index(drop=True)
        
        interview_url = interview_df['url'].iloc[0]
        utt = eval(interview_df['utt'].iloc[0])
        speaker = eval(interview_df['speaker'].iloc[0])
        combined_dialogue = interview_df['combined_dialogue'].iloc[0]

        questions, answers = extract_interviewer_questions(utt, speaker)

        prompts_list = []
        for q_idx, question in enumerate(questions):
            prompt = get_classify_all_questions_taxonomy_prompt(combined_dialogue, question)
            prompt_dict = {
                "custom_id": f"{interview_id}_q{q_idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a world-class annotator for interview questions."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000
                }
            }
            prompts_list.append(prompt_dict)

        output_file = os.path.join(output_dir, f"{interview_id}_schema_classification_prompts.jsonl")
        with open(output_file, "w") as file:
            for prompt in prompts_list:
                file.write(json.dumps(prompt) + "\n")

        logging.info(f"Saved {len(prompts_list)} prompts to {output_file}")

def upload_jsonl_file(file_path):
    batch_input_file = client.files.create(
            file=open(file_path, "rb"),
            purpose="batch"
        )  
    return batch_input_file.id

def create_batch(file_id, model="gpt-4o"):
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Batch processing job for interview question classification"
        }
    )
    return batch.id

def check_batch_status(batch_id):
    batch_status = client.batches.retrieve(batch_id)
    return batch_status

def download_batch_results(batch_id, interview_id, output_dir="output_results/gpt_batching/gpt_type_classify"):
    batch_status = check_batch_status(batch_id)
    output_file_id = batch_status.output_file_id
    
    if output_file_id:
        result_file = client.files.content(output_file_id)
        output_file_path = os.path.join(output_dir, f"{interview_id}_results.jsonl")
        with open(output_file_path, "wb") as f:
            f.write(result_file.read())
        logging.info(f"Results saved to {output_file_path}")
        return output_file_path
    else:
        logging.warning("Batch is not yet complete or no output file available.")
        return None

def gpt_save_results_to_csv(original_df, jsonl_path, csv_output_dir="output_results/gpt_batching/gpt4o_csv_outputs"):
    os.makedirs(csv_output_dir, exist_ok=True)
    interview_id = os.path.splitext(os.path.basename(jsonl_path))[0].split('_')[0]

    relevant_interview_df = original_df[original_df['id'] == interview_id]

    combined_dialogue = relevant_interview_df.iloc[0]['combined_dialogue']
    interview_url = relevant_interview_df.iloc[0]['url']
    utt = eval(relevant_interview_df['utt'].iloc[0])
    speaker = eval(relevant_interview_df['speaker'].iloc[0])
    questions, answers = extract_interviewer_questions(utt, speaker)
    
    csv_output_path = os.path.join(csv_output_dir, f"Interview_{interview_id}_questions_classified.csv")
    data = [{
        "Interview id": interview_id,
        "interview_url": interview_url,
        "combined_dialogue": combined_dialogue,
        "Question (Interviewer)": None,
        "Answer (Guest)": None,
        "Question type": None
    }]
    
    with open(jsonl_path, "r") as file:
        for i, line in enumerate(file):
            result = json.loads(line)
            content = result.get("response", {}).get("body", {}).get("choices", [])[0].get("message", {}).get("content", "")
            question_type = extract_text_inside_brackets(content) if extract_text_inside_brackets(content).lower() in TAXONOMY else f"(MISC) {extract_text_inside_brackets(content)}"
            
            data.append({
                "Interview id": "", 
                "interview_url": "",  
                "combined_dialogue": "",
                "Question (Interviewer)": questions[i] if i < len(questions) else "",
                "Answer (Guest)": answers[i] if i < len(answers) else "", 
                "Question type": question_type
            })

    df = pd.DataFrame(data)
    df.to_csv(csv_output_path, index=False)
    logging.info(f"Results saved to {csv_output_path}")

def extract_interview_ids_from_filenames(filenames):
    return {os.path.splitext(f)[0].split('_')[0] for f in filenames}

def process_jsonl_files(original_df, data_path = "output_results/gpt_batching/batch_prompts", output_dir="output_results/gpt_batching/gpt_type_classify"):
    os.makedirs(output_dir, exist_ok=True)
    batch_files = [f for f in os.listdir(data_path) if f.endswith('.jsonl')]
    all_interview_ids = extract_interview_ids_from_filenames(batch_files)

    processed_files = [f for f in os.listdir(output_dir) if f.endswith('.jsonl')]
    processed_interview_ids = extract_interview_ids_from_filenames(processed_files)

    remaining_interview_ids = all_interview_ids - processed_interview_ids

    for jsonl_file in batch_files:
        
        interview_id = os.path.splitext(jsonl_file)[0].split('_')[0]
        if interview_id not in remaining_interview_ids:
            logging.info(f"Skipping already processed interview ID: {interview_id}")
            continue
        
        file_path = os.path.join(data_path, jsonl_file)
        
        file_id = upload_jsonl_file(file_path)
        logging.info(f"Uploaded file ID: {file_id}")

        batch_id = create_batch(file_id)
        logging.info(f"Created batch ID: {batch_id}")

        while True:
            status = check_batch_status(batch_id)
            logging.info(f"Batch status: {status.status}")
            if status.status == 'completed':
                break
            time.sleep(60)

        result_jsonl_path = download_batch_results(batch_id, interview_id, output_dir)

        if result_jsonl_path:
            gpt_save_results_to_csv(original_df, result_jsonl_path)


if __name__ == "__main__":
    df = pd.read_csv("/project/jonmay_231/spangher/Projects/news-interview-question-generation/dataset/final_dataset.csv")
    # generate_batched_schema_classification_prompts(df, sample_size=150)
    process_jsonl_files(df)

    # to download the csv files, navigate to output_results/gpt_batching/gpt4o_csv_outputs
