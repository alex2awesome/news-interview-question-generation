# vllm-type-classification.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import vllm_infer, extract_text_inside_brackets
from prompts import TAXONOMY, CLASSIFY_USING_TAXONOMY_PROMPT

def classify_question(model_name, messages):
    generated_text = vllm_infer(model_name, messages)
    print(f"generated_text: {generated_text}")
    
    question_type = extract_text_inside_brackets(generated_text)
          
    if question_type in TAXONOMY:
        return question_type
    else:
        return "Unknown question type"

if __name__ == "__main__":
    transcript = """ 

    RACHEL MARTIN, HOST:

    Howard Lutnick is the CEO of the financial firm Cantor Fitzgerald. His company occupied the 101st to 105th floors of One World Trade Center. On September 11, 2001, he lost his brother and 658 of his colleagues. Lutnick survived and vowed to keep the firm alive. Now, 15 years later, he is still the CEO. And he joins us on the line from New York. Thank you so much for taking the time.

    HOWARD LUTNICK: Hey. It's my pleasure, Rachel.

    MARTIN: I'm sure there are a lot of moments and conversations that stand out from that first 24-hour period. But could I ask you to share one or two that stick with you?

    LUTNICK: Sure. So the night of September 11, I didn't really know who was alive and who wasn't alive. So we had a conference call. It was about 10 o'clock at night. And my employees called in. And I said, look, we have two choices.

    We can shut the firm down and go to our friends' funerals. Remember, that would be 20 funerals a day every day for 35 straight days. And I've got to tell you, If'm not really interested in going to work. All I want to do is climb under the covers and hug my family.

    But if we are going to go to work, we're going to do it to take care of our friends' families. So what do you want to do? You guys want to shut it down? Or do you want to work harder than you've ever worked before in your life? And that was the moment where the company survived.

    MARTIN: You weren't there on that morning.
    """
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    prompt = CLASSIFY_USING_TAXONOMY_PROMPT.format(transcript_section=transcript)
    messages = [
        {"role": "system", "content": "You are a world-class annotator for interview questions."},
        {"role": "user", "content": prompt}
    ]
    
    q_type = classify_question(model_name, messages)
    print(f"Outputed Label: {q_type}")