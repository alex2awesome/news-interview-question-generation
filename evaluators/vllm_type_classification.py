# vllm-type-classification.py

import sys
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
from transformers import AutoTokenizer
import gc
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import vllm_infer, vllm_infer_batch, load_vllm_model, extract_text_inside_brackets, stitch_csv_files
from prompts import get_classify_taxonomy_prompt, TAXONOMY
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def type_classification_prompt_loader(QA_seq, question):
    prompt = get_classify_taxonomy_prompt(QA_seq, question)
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
    question_types = [extract_text_inside_brackets(output) if extract_text_inside_brackets(output).lower() in TAXONOMY else f"(MISC) {extract_text_inside_brackets(output)}" for output in outputs]
    return question_types

# this adds a column to LLM_questions_df called LLM_Question_Type and Actual_Question_Type
def classify_question_process_dataset(LLM_questions_df, output_dir="output_results", batch_size=50, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    LLM_question_types_results = []
    Actual_question_types_results = []

    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for start_idx in range(0, len(LLM_questions_df), batch_size):
        batch = LLM_questions_df.iloc[start_idx:start_idx + batch_size]
        
        QA_Sequences = batch['QA_Sequence']
        LLM_questions = batch['LLM_Question']
        Actual_questions = batch['Actual_Question']

        LLM_question_types = classify_question_batch(QA_Sequences, LLM_questions, model, tokenizer)
        Actual_question_types = classify_question_batch(QA_Sequences, Actual_questions, model, tokenizer)

        LLM_question_types_results.extend(LLM_question_types)
        Actual_question_types_results.extend(Actual_question_types)

        gc.collect()

    LLM_questions_df['LLM_Question_Type'] = LLM_question_types_results
    LLM_questions_df['Actual_Question_Type'] = Actual_question_types_results

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'LLM_classified_results.csv')
    LLM_questions_df.to_csv(output_file_path, index=False)
    print(f"csv file saved to {output_file_path}")
    return LLM_questions_df

# implementation 2: save by batch + functionality to start where u stop
def efficient_classify_question_process_dataset(LLM_questions_df, output_dir="output_results", batch_size=50, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    os.makedirs(output_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(output_dir) if re.match(r'LLM_classified_results_\d+_\d+\.csv', f)]
    if existing_files:
        last_file = sorted(existing_files, key=lambda x: int(re.search(r'_(\d+)\.csv', x).group(1)))[-1]
        last_end_idx = int(re.search(r'_(\d+)\.csv', last_file).group(1))
        current_idx = last_end_idx
        print(f"Resuming from index {current_idx}")
    else:
        current_idx = 0
    
    model = load_vllm_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for start_idx in range(current_idx, len(LLM_questions_df), batch_size):
        batch = LLM_questions_df.iloc[start_idx:start_idx + batch_size]
        
        QA_Sequences = batch['QA_Sequence']
        LLM_questions = batch['LLM_Question']
        Actual_questions = batch['Actual_Question']

        LLM_question_types = classify_question_batch(QA_Sequences, LLM_questions, model, tokenizer)
        Actual_question_types = classify_question_batch(QA_Sequences, Actual_questions, model, tokenizer)

        temp_df = batch.copy()
        temp_df['LLM_Question_Type'] = LLM_question_types
        temp_df['Actual_Question_Type'] = Actual_question_types

        output_file_path = os.path.join(output_dir, f'LLM_classified_results_{start_idx}_{start_idx + batch_size}.csv')
        temp_df.to_csv(output_file_path, index=False)
        print(f"Batch {start_idx} to {start_idx + batch_size} saved to {output_file_path}")

        gc.collect()

    print("All batches processed and saved.")
    LLM_classified_df = stitch_csv_files(output_dir, 'final_LLM_classified_results.csv')
    return LLM_classified_df

if __name__ == "__main__":
    # dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/test/QA_Seq_LLM_generated.csv"
    # df = pd.read_csv(dataset_path)

    # new_df = classify_question_process_dataset(df, output_dir="output_results/test/type_classification", batch_size=100, model_name="meta-llama/Meta-Llama-3-8B-Instruct") # saves type_classification labels in LLM_classified_results.csv
    # print(new_df)

    # dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/test/type_classification/LLM_classified_results.csv"
    # df = pd.read_csv(dataset_path)
    # filtered_df = df[(df["LLM_Question_Type"].str.contains("(MISC)", na=False)) | 
    #                  (df["Actual_Question_Type"].str.contains("(MISC)", na=False))
    #                 ]
    # LLM_Question_Type = filtered_df["LLM_Question_Type"].tolist()
    # Actual_Question_Type = filtered_df["Actual_Question_Type"].tolist()
    # count = 0
    # for guess, actual in zip(LLM_Question_Type, Actual_Question_Type):
    #     if "(MISC)" in guess:
    #         count += 1
    #         print(guess)
    #     if "(MISC)" in actual:
    #         count += 1
    #         print(actual)
    # print(f"proportion of the errors in a sample of 300 data points: {count/df.shape[0]}")
 
    # new_df = classify_question_process_dataset(df, output_dir="output_results/test/type_classification", model_name="meta-llama/Meta-Llama-3-8B-Instruct") # saves type_classification labels in LLM_classified_results.csv
    # print(new_df)


    QA_seq = '''
    FARAI CHIDEYA, host: I'm Farai Chideya, and this is NEWS & NOTES. It's time now for our Africa update. Today, sisters are doing it for themselves. We're talking about the women leaders of Africa. Yesterday, President Bush awarded the Presidential Medal of Freedom to seven people including Liberian President Ellen Johnson-Sirleaf. She is the first woman elected to head an African country. President GEORGE W. BUSH: All her life, President Sirleaf has been a pioneer. The daughter of a school teacher in Monrovia, she crossed the ocean as a young woman and earned three degrees in the United States. She's been a business executive, a development expert, a public official and always a patriot. She loves Liberia, and she loves all its people. So how much progress have women made in leadership across the continent? We've got Emira Woods to give us some perspective. She is co-director of Foreign Policy in Focus at the Institute of Policies Studies. Hi, Emira. 
    Ms. EMIRA WOODS (Co-director, Foreign Policy in Focus, Institute of Policy Studies): How are you, Farai? 
    FARAI CHIDEYA, host: I'm doing great. So you're from Liberia. What does this honor mean for your country? 
    Ms. EMIRA WOODS (Co-director, Foreign Policy in Focus, Institute of Policy Studies): Well, it is extraordinary that Liberia, after a history of 26 years of war, emerges with Africa's first woman president. And we are really happy to have, not only the Presidential Medal of Freedom awarded to Ellen Johnson-Sirleaf, but, you know, of high distinction of addressing both members of Congress and a number of distinctions coming her way. So it's exciting. But it is, Farai, really, on the symbolic end. And what we're hoping for as Liberians and member of African community overall is that these symbolic measures are actually backed up by concrete actions on debt cancellation for Liberia and on other real concrete steps that the Bush administration can take to actually impact women's lives in Liberia and around the continent. 
    FARAI CHIDEYA, host: Now, before Ellen Johnson-Sirleaf was elected president in 2006, how was the balance of power in terms of gender? Were there a lot of female elected officials or female leaders in Liberia? 
    Ms. EMIRA WOODS (Co-director, Foreign Policy in Focus, Institute of Policy Studies): Well, you know, Liberia has a mixed history. Back in the 1960s, Liberia was actually one of the first countries to send a woman to the United Nations as United Nations ambassador, so by - in, like, 1967, '68, Liberia was represented by a woman at the U.N. So there have been some incredible instances of women's political leadership and there - those have also followed in civil society throughout Liberia. There have been instances of women leaders. However, over the past years, it has been men dominating and, in particular, the Samuel Does, the Charles Taylors - some really ruthless men dominating the political scene in Liberia. 
    FARAI CHIDEYA, host: Is there a program akin to affirmative action in the U.S. that deals with any of these gender issues? 
    Ms. EMIRA WOODS (Co-director, Foreign Policy in Focus, Institute of Policy Studies): Well, there is beginning to be discussion of that, and it is happening in some ways in Liberia both looking at the legislator and changes that can be put in place to help bring about a greater participation of women in political office. But that's just beginning in Liberia. I think there are many other countries that are much further along in terms of affirmative action through constitutional changes and other party political changes that have been taking place. 
    FARAI CHIDEYA, host: Well, let's move on to Rwanda. It's a nation that's rebuilding itself after genocide. And now, women make up nearly half of parliament. That is more than any other parliamentary body in the world. What's behind the trend? 
    Ms. EMIRA WOODS (Co-director, Foreign Policy in Focus, Institute of Policy Studies): It is absolutely extraordinary to see what has happened. Most people assumed that, you know, the men were killed in the genocide. But if you look at the numbers, actually, the population of women is about 50 percent. And the population of women reflected in parliament is about 50 percent. So it isn't reflective of any dramatic changes in the population. But it is women's political leadership at all levels that's emerging because there was a commitment made right after the genocide, essentially, to change the constitution, to change the political party structures, to put in place concrete quotas for women. And so you see at every level - from municipal to regional to national elections - you see a space being opened for women. Thirty percent in many cases, women mandated in these posts as candidates and it carries forward that as women are learning leadership roles at those local levels, they continue to advance their leadership all the way through to the national level. And this is what you see reflected with 50 percent of their parliament controlled by women. 
    FARAI CHIDEYA, host: Are there any specific ways in which women have helped shape the destiny of Rwanda after the war? 
    Ms. EMIRA WOODS (Co-director, Foreign Policy in Focus, Institute of Policy Studies): Well, that's been the enormous potential here of women's political leadership. If you see the research that's been done on Rwanda, in particular, indicates that there are changes in women's impact in broader processes. So there is greater funding for health, for education. Those essential services needed to have healthy, sustainable lives. You see a greater amount of the national budget being directed at those essential services. So, you know, I think when we look at Rwanda and other countries like Rwanda that have made this commitment to women's political leadership, we see the potential of women, not only impacting the political processes, but impacting the very nature of people's lives at the household level through these policies that are being put in place. 
    '''
    question = "FARAI CHIDEYA, host: Now, moving on again to South Africa. There's another rise in women leaders. There's also been some legislation that has allowed that to happen or encouraged that to happen. Give us a sense of that."
    print(get_classify_taxonomy_prompt(QA_seq, question))

    # expected result: dataframe now contains the following columns: QA_Sequence, Actual_Question, LLM_Question, LLM_Question_Type, Actual_Question_Type
