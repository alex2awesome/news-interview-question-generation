# LLM-question-generation.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import vllm_infer, extract_text_inside_brackets, extract_text_inside_parentheses
from prompts import CONTEXT_GENERATOR_PROMPT, LLM_QUESTION_GENERATOR_PROMPT

def LLM_question_gen_prompt_loader(QA_seq):
    prompt = LLM_QUESTION_GENERATOR_PROMPT.format(QA_Sequence=QA_seq)
    messages = [
        {"role": "system", "content": "You are a world-class interview question guesser."},
        {"role": "user", "content": prompt}
    ]
    return messages

def LLM_question_generator(messages, model_name="meta-llama/Meta-Llama-3-70B-Instruct"):
    generated_text = vllm_infer(messages, model_name)
    print(f"generated_text: {generated_text}")
    LLM_question = extract_text_inside_brackets(generated_text)
    motivation = extract_text_inside_parentheses(generated_text)

    return LLM_question, motivation


if __name__ == "__main__":
    # transcript = """ 
    # RACHEL MARTIN, HOST:

    # Howard Lutnick is the CEO of the financial firm Cantor Fitzgerald. His company occupied the 101st to 105th floors of One World Trade Center. On September 11, 2001, he lost his brother and 658 of his colleagues. Lutnick survived and vowed to keep the firm alive. Now, 15 years later, he is still the CEO. And he joins us on the line from New York. Thank you so much for taking the time.

    # HOWARD LUTNICK: Hey. It's my pleasure, Rachel.

    # MARTIN: I'm sure there are a lot of moments and conversations that stand out from that first 24-hour period. But could I ask you to share one or two that stick with you?

    # LUTNICK: Sure. So the night of September 11, I didn't really know who was alive and who wasn't alive. So we had a conference call. It was about 10 o'clock at night. And my employees called in. And I said, look, we have two choices.

    # We can shut the firm down and go to our friends' funerals. Remember, that would be 20 funerals a day every day for 35 straight days. And I've got to tell you, If'm not really interested in going to work. All I want to do is climb under the covers and hug my family.

    # But if we are going to go to work, we're going to do it to take care of our friends' families. So what do you want to do? You guys want to shut it down? Or do you want to work harder than you've ever worked before in your life? And that was the moment where the company survived.

    # MARTIN: You weren't there on that morning.

    # LUTNICK: So it was my son Kyle's first day of kindergarten. And then as I walk him upstairs, an administrator grabs me and says, you know, your office is looking for you. A plane has hit the building. So I jumped in the car. And obviously, as I got down there, I saw that huge, black, billowing smoke.

    # And the guy that was driving my car started crying. I was like, let's just - we got to get there. We just got to get there. And so I drove right to the building and got to the door of the building and started grabbing people as they came out.

    # MARTIN: Did you see anyone you knew?

    # LUTNICK: I didn't. You know, I was grabbing people as they came out. And then I heard this sound. It was the loudest sound I'd ever heard. What it was was Two World Trade Center collapsed. So I just start running. I'm just running.

    # And I look over my shoulder. And there's this big, giant, black tornado of smoke chasing me. So I dive under a car. And the black smoke just went foosh (ph). You know, I was saying, don't breathe, don't breathe, don't breathe. And then take a deep breath.

    # And then, you know, all this sort of particles and soot and dust starts filling my lungs. I understood right away that if I was suffocating and I was outside, how could they possibly have survived inside?

    # MARTIN: Was your brother in the building?

    # LUTNICK: My brother Gary was 36, and he was in the building. And later that night when I spoke to my sister, she told me that she spoke to my brother. And she had said to him, oh, my God. Thank God you're not there. You know, thank God you're not there.

    # And he said, I am here. And I'm going to die. And I just wanted to tell you I love you. And he said goodbye. He said - you know, my sister got to talk to my brother when he said goodbye. I still get choked up. Sorry.

    # MARTIN: That's OK.

    # So you had a business to run amidst overwhelming grief. How did you begin to put those pieces together?

    # LUTNICK: Well, decisions I made were they needed to have a boss for the business. If I didn't have a leader, I shut it. And I had a division of 86 people where four people survived. And you can't really build a business back with four people.

    # Basically, we went from being a great company that was making a million dollars a day to a company that was losing a million dollars a day. But they all have mortgages to pay, and they need to put food on their table.

    # So one of the things I did is I would call the leaders of other companies and say, here, this guy's John. He sells this many products. He's incredibly successful. You would've never been able to hire him. He was never going to come work for you.

    # MARTIN: (Laughter).

    # LUTNICK: But here's what I need. I need you to match what he made with me. And if not, in one hour, I'm going to call your other competitor. And he's going to be working there. And I got all my people jobs.

    # And then, number two, I sat down with my sister. So she is sort of curled up in a ball in the corner. And I said, Edie, I need you to run the relief fund. I need to take care of these families. I need someone to tell that we care. And I have to try to rebuild the company so I can send them some money. She's like, I can't do it. I can't do it. And she's, like, sickened.

    # And I said, no, no, you have to do it. She had a completely, totally destroyed and broken heart. And when you need to talk to other people who have a destroyed, broken heart, you know, if you don't have one, your voice sounds like tin.

    # MARTIN: Did it change what kind of leader you became?

    # LUTNICK: Well, I had a - I'd been to hell before on September 12, 1979, when my dad got killed. And my extended family - they pulled out instead of coming in. And I was not going to repeat that in my life. And so the drive, for me, was to show that I'm a human being.

    # MARTIN: What do you do on September 11 every year?

    # LUTNICK: Well, this year, we'll have a memorial. But the closest business day - so in this case, it's Monday the 12 - I ask all my employees to waive their day's pay. And you can't make people work their tail off and not pay them. It doesn't really work that way.

    # But they all waive their day's pay. And then every penny of revenue that comes in the door, we give away. Last year, we raised $12 million that day. So they go home, and they think 9/11 is a beautiful thing. And it binds my company together. It lets the new people understand who we are and what rages inside of our soul and that we're always going to bring 9/11 - it's a part of us, but it doesn't define us. But it is us.

    # MARTIN: Howard Lutnick is the CEO of Cantor Fitzgerald. Mr. Lutnick, thank you so much for sharing your thoughts this weekend.

    # LUTNICK: All right. Thanks, Rachel.
    # """
    
    QA_seq = """ 

    RACHEL MARTIN, HOST:

    Howard Lutnick is the CEO of the financial firm Cantor Fitzgerald. His company occupied the 101st to 105th floors of One World Trade Center. On September 11, 2001, he lost his brother and 658 of his colleagues. Lutnick survived and vowed to keep the firm alive. Now, 15 years later, he is still the CEO. And he joins us on the line from New York. Thank you so much for taking the time.

    HOWARD LUTNICK: Hey. It's my pleasure, Rachel.

    MARTIN: I'm sure there are a lot of moments and conversations that stand out from that first 24-hour period. But could I ask you to share one or two that stick with you?

    LUTNICK: Sure. So the night of September 11, I didn't really know who was alive and who wasn't alive. So we had a conference call. It was about 10 o'clock at night. And my employees called in. And I said, look, we have two choices.

    We can shut the firm down and go to our friends' funerals. Remember, that would be 20 funerals a day every day for 35 straight days. And I've got to tell you, If'm not really interested in going to work. All I want to do is climb under the covers and hug my family.

    But if we are going to go to work, we're going to do it to take care of our friends' families. So what do you want to do? You guys want to shut it down? Or do you want to work harder than you've ever worked before in your life? And that was the moment where the company survived.

    MARTIN: You weren't there on that morning.
    """
    # prompt1 = CONTEXT_GENERATOR_PROMPT.format(transcript=transcript)
    # messages1 = [
    #     {"role": "system", "content": "You are a world-class interview transcript synthesizer."},
    #     {"role": "user", "content": prompt1}
    # ]
    
    # context = vllm_infer(model_name, messages1)
    # prompt2 = LLM_QUESTION_GENERATOR_PROMPT.format(QA_Sequence=QA_Sequence)
    # messages2 = [
    #     {"role": "system", "content": "You are a world-class interview question guesser."},
    #     {"role": "user", "content": f"{context} {prompt2}"}
    # ]

    messages = LLM_question_gen_prompt_loader(QA_seq)
    LLM_question, motivation = LLM_question_generator(messages, "meta-llama/Meta-Llama-3-8B-Instruct")
    print(f'LLM Generated Question: {LLM_question}')
    print(f'Motivation For Generated Question: {motivation}')