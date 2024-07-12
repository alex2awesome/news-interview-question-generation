# LLM-question-generation.py

import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import vllm_infer, extract_text_inside_brackets, extract_text_inside_parentheses, create_combined_dialogue_df
from prompts import CONTEXT_GENERATOR_PROMPT, LLM_QUESTION_GENERATOR_PROMPT

def LLM_question_gen_prompt_loader(QA_seq):
    prompt = LLM_QUESTION_GENERATOR_PROMPT.format(QA_Sequence=QA_seq)
    messages = [
        {"role": "system", "content": "You are a world-class interview question guesser."},
        {"role": "user", "content": prompt}
    ]
    return messages

def LLM_question_generator(QA_seq, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    messages = LLM_question_gen_prompt_loader(QA_seq)
    generated_text = vllm_infer(messages, model_name)
    print(f"generated_text: {generated_text}")
    LLM_question = extract_text_inside_brackets(generated_text)
    motivation = extract_text_inside_parentheses(generated_text)

    return LLM_question, motivation

# reformats dataset transcript --> QA_sequence, feeds each QA_Seq into LLM to predict next question, saves prediction
def LLM_question_process_dataset(file_path, output_dir="output_results"):
    dataset = create_combined_dialogue_df(file_path, output_dir)
    question_results = []
    motivation_results = []

    for _, row in dataset.iterrows():
        QA_seq = row['combined_dialogue']
        LLM_question, motivation = LLM_question_generator(QA_seq, "meta-llama/Meta-Llama-3-8B-Instruct")
        question_results.append(LLM_question)
        motivation_results.append(motivation)

    results_df = pd.DataFrame({
        'QA_Sequence': dataset['combined_dialogue'],
        'Generated_Question': question_results,
        'Motivation': motivation_results
    })

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'LLM_generated_results.csv')
    results_df.to_csv(output_file_path, index=False)

# def LLM_question_generator(QA_seq, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
#     return "placeholder question?", "test_blablabla"

if __name__ == "__main__": 
    dataset_path = os.path.join("dataset", "combined_data.csv")
    LLM_question_process_dataset(dataset_path)
    df = pd.read_csv(os.path.join("output_results", "LLM_generated_results.csv"))
    print(df.head())

    # use for testing
    # example_QA_seq = """ 

    # RACHEL MARTIN, HOST:

    # Howard Lutnick is the CEO of the financial firm Cantor Fitzgerald. His company occupied the 101st to 105th floors of One World Trade Center. On September 11, 2001, he lost his brother and 658 of his colleagues. Lutnick survived and vowed to keep the firm alive. Now, 15 years later, he is still the CEO. And he joins us on the line from New York. Thank you so much for taking the time.

    # HOWARD LUTNICK: Hey. It's my pleasure, Rachel.

    # MARTIN: I'm sure there are a lot of moments and conversations that stand out from that first 24-hour period. But could I ask you to share one or two that stick with you?

    # LUTNICK: Sure. So the night of September 11, I didn't really know who was alive and who wasn't alive. So we had a conference call. It was about 10 o'clock at night. And my employees called in. And I said, look, we have two choices.

    # We can shut the firm down and go to our friends' funerals. Remember, that would be 20 funerals a day every day for 35 straight days. And I've got to tell you, If'm not really interested in going to work. All I want to do is climb under the covers and hug my family.

    # But if we are going to go to work, we're going to do it to take care of our friends' families. So what do you want to do? You guys want to shut it down? Or do you want to work harder than you've ever worked before in your life? And that was the moment where the company survived.

    # MARTIN: You weren't there on that morning.
    # """

    # LLM_question, motivation = LLM_question_generator(example_QA_seq, "meta-llama/Meta-Llama-3-8B-Instruct")
    # print(f'LLM Generated Question: {LLM_question}')
    # print(f'Motivation For Generated Question: {motivation}')
