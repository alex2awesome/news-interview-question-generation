# vllm-consistency-eval.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import vllm_infer, extract_text_inside_brackets
from prompts import DIMENSION_OF_SIMILARITY_PROMPT
# from vllm-type-classification import classify_question 
# ^include classify_question(model_name, messages) as a parameter later

def consistency_eval_prompt_loader(transcript_context, llm_question, human_question):
    prompt = DIMENSION_OF_SIMILARITY_PROMPT.format(
        transcript_context=transcript_context,
        LLM_question=llm_question,
        human_question=human_question
    )
    messages = [
        {"role": "system", "content": "You are a world-class annotator for question similarity."},
        {"role": "user", "content": prompt}
    ]
    return messages

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

if __name__ == "__main__":
    ex_llm_question = "What are the main causes of climate change?"
    ex_human_question = "Can you explain why the climate is changing?"
    ex_transcript_context = "We are discussing environmental issues, particularly focusing on climate change and its causes."
    messages = consistency_eval_prompt_loader(ex_transcript_context, ex_llm_question, ex_human_question)
    
    sim_score = consistency_compare(messages, "meta-llama/Meta-Llama-3-8B-Instruct")
    print(f'Total similarity score: {sim_score}')

    '''
    Future implementation: 
    1. Incorporate function call to classify_question(model_name, messages) 
    and include it as an input to make the LLM evalution function more robust.
    2. Set up a function that is capable of loading entire batch of data.
    '''