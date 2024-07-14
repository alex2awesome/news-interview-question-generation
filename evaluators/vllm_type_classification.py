# vllm-type-classification.py

import sys
import os
import pandas as pd
from transformers import AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import vllm_infer, vllm_infer_batch, load_vllm_model, extract_text_inside_brackets, create_QA_Sequence_df_N_qa_pairs
from prompts import TAXONOMY, CLASSIFY_USING_TAXONOMY_PROMPT
from LLM_question_generation import LLM_question_process_dataset
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def type_classification_prompt_loader(QA_seq, question):
    prompt = CLASSIFY_USING_TAXONOMY_PROMPT.format(transcript_section=QA_seq, question=question)
    messages = [
        {"role": "system", "content": "You are a world-class annotator for interview questions."},
        {"role": "user", "content": prompt}
    ]
    return messages
# single-use
def classify_question(QA_Sequence, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    messages = type_classification_prompt_loader(QA_Sequence)
    generated_text = vllm_infer(messages, model_name)
    print(f"generated_text: {generated_text}")
    
    question_type = extract_text_inside_brackets(generated_text)
          
    if question_type in TAXONOMY:
        return question_type
    else:
        return "Unknown question type"

# for batching
def classify_question_batch(QA_Sequences, questions, model, tokenizer):
    messages_batch = [type_classification_prompt_loader(QA_seq, question) for QA_seq, question in zip(QA_Sequences, questions)]
    formatted_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    outputs = vllm_infer_batch(formatted_prompts, model)
    question_types = [extract_text_inside_brackets(output) if extract_text_inside_brackets(output) in TAXONOMY else "Error" for output in outputs]
    return question_types

# this adds a column to LLM_questions_df called LLM_Question_Type and Actual_Question_Type
def classify_question_process_dataset(LLM_questions_df, output_dir="output_results", batch_size=100, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    LLM_question_types_results = []
    Actual_question_types_results = []

    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for start_idx in range(0, len(LLM_questions_df), batch_size):
        batch = LLM_questions_df.iloc[start_idx:start_idx + batch_size]
        
        QA_Sequences = batch['QA_Sequence'].tolist()
        LLM_questions = batch['LLM_Question'].tolist()
        Actual_questions = batch['Actual_Question'].tolist()

        LLM_question_types = classify_question_batch(QA_Sequences, LLM_questions, model, tokenizer)
        Actual_question_types = classify_question_batch(QA_Sequences, Actual_questions, model, tokenizer)

        LLM_question_types_results.extend(LLM_question_types)
        Actual_question_types_results.extend(Actual_question_types)

    LLM_questions_df['LLM_Question_Type'] = LLM_question_types_results
    LLM_questions_df['Actual_Question_Type'] = Actual_question_types_results

    output_file_path = os.path.join(output_dir, 'LLM_classified_results.csv')
    os.makedirs(output_dir, exist_ok=True)
    LLM_questions_df.to_csv(output_file_path, index=False)
    return LLM_questions_df

if __name__ == "__main__":
    df = pd.read_csv(os.path.join("output_results", "QA_Seq_LLM_generated.csv"))
    df = classify_question_process_dataset(df, model_name="meta-llama/Meta-Llama-3-8B-Instruct") # adds a column to LLM_questions_df called LLM_Question_Type and Actual_Question_Type
    print(df.head())
    # expected result, contains the following columns: QA_Sequence, Actual_Question, LLM_Question, LLM_Question_Type, Actual_Question_Type




    # example_transcript = """ 
    # RACHEL MARTIN, HOST:
    # Howard Lutnick is the CEO of the financial firm Cantor Fitzgerald. His company occupied the 101st to 105th floors of One World Trade Center. On September 11, 2001, he lost his brother and 658 of his colleagues. Lutnick survived and vowed to keep the firm alive. Now, 15 years later, he is still the CEO. And he joins us on the line from New York. Thank you so much for taking the time.
    # HOWARD LUTNICK: Hey. It's my pleasure, Rachel.
    # MARTIN: I'm sure there are a lot of moments and conversations that stand out from that first 24-hour period. But could I ask you to share one or two that stick with you?
    # LUTNICK: Sure. So the night of September 11, I didn't really know who was alive and who wasn't alive. So we had a conference call. It was about 10 o'clock at night. And my employees called in. And I said, look, we have two choices.
    # We can shut the firm down and go to our friends' funerals. Remember, that would be 20 funerals a day every day for 35 straight days. And I've got to tell you, If'm not really interested in going to work. All I want to do is climb under the covers and hug my family.
    # But if we are going to go to work, we're going to do it to take care of our friends' families. So what do you want to do? You guys want to shut it down? Or do you want to work harder than you've ever worked before in your life? And that was the moment where the company survived.
    # MARTIN: You weren't there on that morning.
    # """

    # question_type = classify_question(example_transcript, "meta-llama/Meta-Llama-3-8B-Instruct")
    # print(f"Outputed Label: {question_type}")
    # example output: "Outputed Label: Verification Questions - Confirmation"