from openai import OpenAI
import re
import os
from ..prompts import DIMENSION_OF_SIMILARITY_PROMPT
from helper_functions.py import extract_text_inside_brackets

openai_client = get_openai_client()

def consistency_compare(LLM_question, human_question, transcript_context):
    completion = openai_client.chat.completions.create(
          model="gpt-4o",
          messages=[
                {"role": "system", "content": "You are a world-class annotator for question similarity."},
                {"role": "user", "content": DIMENSION_OF_SIMILARITY_PROMPT}
          ]
    )

    generated_text = completion.choices[0].message.content.strip()
    print(f'{generated_text}\n\n')
    
    similarity_scores_str = extract_text_inside_brackets(generated_text)
    similarity_scores_list = similarity_scores_str.split(', ')
    print(f'new sim score: {similarity_scores_list}')
    similarity_scores = [1 if score.lower() == 'yes' else 0 for score in similarity_scores_list]
    print(f'old sim score: {similarity_scores}')
    
    def similarity_score(scores): 
        return sum(scores)
    
    return similarity_score(similarity_scores)

if __name__ == "__main__":
    llm_question = "What are the main causes of climate change?"
    human_question = "Can you explain why the climate is changing?"
    transcript_context = "We are discussing environmental issues, particularly focusing on climate change and its causes."
    print(f'Total similarity score: {consistency_compare(llm_question, human_question, transcript_context)}')