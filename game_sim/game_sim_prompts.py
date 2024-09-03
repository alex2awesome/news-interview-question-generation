# prompt/instructions for the interviewee
def get_source_prompt(QA_Sequence, info_items, response_type = "honest"):
    prompt = f'''
    You are a source. You have the following pieces of information but you are not just going to say it all at once. Only answer properly if any of your information items match the journalist’s question:

    Here are your information items:

    {info_items}
    
    Here is the conversation so far:

    {QA_Sequence}

    Decide if the last question the interviewer asked aligns with any of your information items. If so, respond and make sure you are {response_type}. 
    If the question doesn't align with any of the information items you have, then say something like: "I'm not sure." of "I'll get back to you on that."

    Now, reply to the question and wrap your answer in brackets. 
    
    Format: 
        Can question be answered by informations I have?
            if yes: 
                The question asked by the interviewer can be answered by an information item I have. 
                I will answer with information item <insert information item number>:
                Here is my response to the question: 
                [<response>]
            if no:  
                The question asked by the interviewer cannot be answered by an information items I have.
                Here is my response to the question: 
                [<response>]
    
    Here are some examples:
        
        Example 1:
            The question asked by the interviewer can be answered by an information item I have. 
            I will answer with information item 5:
            Here is my response to the question: 
            [Yeah, you know, it's hard to believe that we're talking about Texas as a wind power. 
                But actually Texas ranks as the country's number-two supplier of electricity from wind. It is right behind California. 
                It already has 16 wind farms operating around the state. There's another five or six on the drawing boards. 
                They're going into service this year. And as we know how Texans like to do things in a big way, the wind turbines in Texas are also among the biggest in the country, some standing higher than the Statue of Liberty.]

        Example 2:
            The question asked by the interviewer cannot be answered by an information items I have. 
            Here is my response to the question: 
            [That's a good question, I'm not too sure about this matter.]

    '''
    return prompt

    # for response type you might really need to define what each means
    # make sure the source profile isn't just 100% being evasive, or confused, that's not useful. We want to make sure it's realistic, even if that means we do some rule-based behaviors. maybe we want some set of frequency of specified behavior - eg. response_type = evasive --> 40% evasive, 60% straightforward.
    # i.e youre super anxious, you will only respond honestly after a lot of acknowledgment, etc.

# prompt/instructions for interviewer
def get_interviewer_prompt(QA_Sequence, outline_objectives, strategy = "straightforward"):
    prompt = f'''
    You are an interviewer. Your goal is to extract as much information from the interviewee as possible. 
    
    Here is the outline of objectives you've prepared before the interview:

    {outline_objectives}

    Here is the conversation so far. (If you don't see a conversation, kick the interview off with a starting remark):

    {QA_Sequence}

    Now, ask the next question (be {strategy}) and wrap it with brackets. Format: Here is my next question: [<your response>]
    
    Here are some examples:
        Example 1:
        
        Here is my question: 
        [
            The Syrian government has denounced the weekend evacuation of some rescue workers from the U.S.-backed Syrian Civil Defense group. 
            They're known as the White Helmets, and their mission is to save civilian lives during Syria's civil war. 
            Those who were rescued, along with some of their family members, were transported from Syria to Jordan. Joining us now via Skype is Ibrahim Olabi. 
            He's a human rights lawyer who's worked with the White Helmets. Good morning, Mr. Olabi.
        ]

        Example 2:

        Here is my question: 
        [And what are you hearing from the folks who've been evacuated?]
    '''
    return prompt

OUTLINE_FORMAT = '''
The format of your response should be in this sequence:
  1. First, explain your thought process step by step: 
    - What were the central topics of discussion?
    - What specific areas of the subject's background, expertise, or experiences were explored?
    - Were there any recurring themes or questions that seemed to guide the conversation?
    - How did the interviewer probe for deeper insights or follow up on key points?
  2. Now putting this together, in brackets, create an outline that could have served as the interviewer's guide, with 4-6 broad themes or objectives that are directly relevant to the content of the transcript. Do not simply restate parts of the transcript; instead, synthesize the information into coherent, high-level themes that would shape the flow of the interview.
  3. In other words, place the entire outline you generate in brackets: 
    ie. Here is the format of the generated outline: 
        [
            introductory blurb of the source

            - Objective/Theme 1
            - Objective/Theme 2
            - Objective/Theme 3
            ...
        ]
'''

# generating high-level objectives or topics that can naturally lead to the extraction of the key information without directly giving away those details
def get_outline_prompt(QA_Sequence):
    prompt = f'''
    You are an experienced interviewer preparing for an interview. The goal of this task is to reverse engineer and generate a high-level outline of objectives and general themes that a human interviewer might have prepared before conducting this specific interview.
    You are provided with the complete transcript of the interview. Based on this transcript, identify the key themes, topics, and objectives that the interviewer likely focused on. The outline should be flexible and tailored to the specific content of the interview, reflecting the natural flow and transitions that occurred during the conversation.

    {OUTLINE_FORMAT}

    Here is the interview transcript for your reference:

    {QA_Sequence}

    Please generate the tailored outline in brackets based on this transcript.


    '''
    return prompt

# prompt that summarizes the interview transcript into key information items that the source has
def get_info_items_prompt(QA_Sequence):
    prompt = f"""
    You are tasked with extracting key pieces of information from an interview transcript. 
    
    Below is the transcript:

    {QA_Sequence}

    Please extract the key pieces of information provided by the interviewee, formatted as follows:
    - Information item #1
    - Information item #2
    - Information item #3
    …
    """
    return prompt

# only for topic-transition extraction
def get_all_topic_transition_questions_prompt(QA_Sequence, question):
    prompt = f'''
    I am trying to classify whether certain questions asked by journalists during an interview are topic-transition questions. Topic-transition questions are typically prepared in advance as they shift the conversation from one subject to another.

    Below, I will provide you with the interview transcript for context, followed by the specific question that needs classification.

    Definition of a Topic-Transition Question:
    - Shifts the conversation from one subject (topic A) to a different subject (topic B).
    - Often introduces new topics into the interview.
    - Indicative of outline-level goals in the interview.

    Examples of Topic-Transition Questions:
      1. Previous Question Context: Introduction of the interviewee and their background.
        - Question: "Now I want to talk about Syria. Can you explain how your work in Aleppo changed your career?"
        - Reasoning: The question shifts from the introduction (topic A) to Syria and the interviewee’s work there (topic B).
        - Classification: [Yes]

      2. Previous Question Context: Discussion of the presidential debate.
        - Question: "Let's look forward to the vice-presidential debate. Do you think they will echo what their running mates have been saying?"
        - Reasoning: The topic shifts from the presidential debate (topic A) to the vice-presidential debate (topic B).
        - Classification: [Yes]

    The format of your response should be in this sequence:
      1. Reasoning: First, explain your thought process step by step.
        - How does the given question relate to the previous question? 
        - How does the given question impact the flow of everything before it?
        - Does this question follow in the same overall topic as the previous question/remark or does it start a new topic? 
      2. Then, pick from the following two labels: [yes] or [no]
      3. Classification: Finally, return your guess of the question type, in brackets. i.e. [yes]
    Don't include anything else inside the brackets.
    
    Now it's your turn.

    Interview Transcript:
    {QA_Sequence}

    Given the interview transcript above, please classify the following question as a topic-transition question or not:

    Question: {question}

    Reasoning:

    Classification: 
    '''
    return prompt