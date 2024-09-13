# prompt/instructions for the interviewee
def get_source_specific_info_item(QA_Sequence, info_items):
    prompt = f'''
    You are a source getting interviewed. You have the following pieces of information:
    
    {info_items}

    Here is the interview conversation so far:

    {QA_Sequence}

    Decide if any of the information items answer the last question posed by the interviewer. If so, return which information item you think best aligns with the question in brackets. 
    If the question doesn't align with any of the information items you have, don't respond in brackets. 

    Here are some examples of correct responses:

    Example 1:
        The question asked by the interviewer can be answered by an information item I have: 
            [Information Item #3]
    Example 2:
        The question asked by the interviewer cannot be answered by an information item I have:
            [No information items align with the question]
    '''
    return prompt

# prompt/instructions for the interviewee
def get_source_prompt_basic(QA_Sequence, info_items, specific_info_item, response_type = "honest"):
    prompt = f'''
    You are a source getting interviewed. You have the following pieces of information:

    As a source, here are your information items below. These information items represent the information you possess and can divulge in an interview.

    {info_items}
    
    Here is the conversation so far:

    {QA_Sequence}

    Please use the following information item to craft a natural, conversational response to the interviewer: {specific_info_item}.
    If it says "no information items align with the question", then say something like: "I'm not sure.", or "I'll get back to you on that."

    Now, reply to the question and wrap only the dialogue response in brackets. Below are some examples, and your response should follow its format:
    
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

def get_source_prompt_intermediate(QA_Sequence, info_items, random_segments, response_type = "honest"):
    prompt = f'''
    You are a source getting interviewed. You have the following pieces of information:

    As a source, here are your information items below. These information items represent the information you possess and can divulge in an interview.

    {info_items}
    
    Here is the conversation so far:

    {QA_Sequence}

    Please use the following pieces of information to craft a natural, conversational response to the interviewer: 

    {random_segments}

    If it says "no information items align with the question", then say something like: "I'm not sure.", or "I'll get back to you on that."

    Now, reply to the question and wrap only the dialogue response in brackets. Below are some examples, and your response should follow its format:
    
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

def get_source_starting_prompt(QA_Sequence, info_items):
    prompt = f'''
    You are a source getting interviewed. You have the following pieces of information:

    As a source, here are your information items below. These information items represent the information you possess and can divulge in an interview.

    {info_items}
    
    Here is the conversation so far:

    {QA_Sequence}

    It's the beginning of the interview. Please respond naturally to the interviewer's starting remark. 
    Make sure to write your final response inside brackets. Below are some examples, and your response should follow its format:

    Example 1:
        Here is my response to the starting remark:
        [Thanks for having me on.] 
    Example 2:
        Here is my response to the starting remark: 
        [Thank you for having me.]
    '''
    return prompt

def get_source_ending_prompt(QA_Sequence, info_items):
    prompt = f'''
    You are a source getting interviewed. You have the following pieces of information:

    As a source, here are your information items below. These information items represent the information you possess and can divulge in an interview.

    {info_items}
    
    Here is the conversation so far:

    {QA_Sequence}

    It's the end of the interview. No need to use the information items anymore. Please respond naturally to the interviewer's ending remark. 
    Make sure to write your final response inside brackets. Below are some examples, and your response should follow its format:

    Example 1:
        Here is my response to the ending remark:
        [Thank you.] 
    Example 2:
        Here is my response to the ending remark: 
        [My pleasure. Thank you.]
    '''
    return prompt

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
        [The Syrian government has denounced the weekend evacuation of some rescue workers from the U.S.-backed Syrian Civil Defense group. 
            They're known as the White Helmets, and their mission is to save civilian lives during Syria's civil war. 
            Those who were rescued, along with some of their family members, were transported from Syria to Jordan. Joining us now via Skype is Ibrahim Olabi. 
            He's a human rights lawyer who's worked with the White Helmets. Good morning, Mr. Olabi.]

        Example 2:

        Here is my question: 
        [And what are you hearing from the folks who've been evacuated?]
    
        Make sure only your question is wrapped in brackets.
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

def get_interviewer_starting_prompt(outline_objectives, strategy = "straightforward"):
    prompt = f'''
    You are an interviewer. Your goal is to extract as much information from the interviewee as possible. 
    
    Here is the outline of objectives you've prepared before the interview:

    {outline_objectives}

    You are about to start the interview. Please kick it off with a starting remark. Be {strategy}

    Wrap your starting remark/introduction with brackets. Format: Here is my starting remark: [<your response>]
    
    Here are some examples:
        Example 1:
        
        Here is my starting remark: 
        [We're going to turn now to Siegfried Hecker. He is a nuclear scientist who has been tracking the nuclear 
        program in North Korea for decades. He's seen the country's nuclear facilities firsthand. 
        He's now an emeritus professor at Stanford University, and he sees some promising signs in relations 
        between the U.S. and North Korea. Welcome.]

        Example 2:

        Here is my starting remark: 
        [Football is getting harder to watch even for some of the sport's most passionate fans. Research has shown again and again that the hits those players take can have a lasting impact on the players' brains. The NFL announced this past week that it will spend $100 million to advance concussion research. Some of that money will go into continuing efforts to develop a safer helmet. Doctors say so far, helmets have done little to reduce concussions and the long-term effects of repeated head trauma. Joining us now to talk about this is Dr. David Camarillo. He's assistant professor of bioengineering at Stanford and he leads a lab dedicated to inventing equipment that reduces traumatic brain injury in sports. Welcome to the program.]
    
        Make sure only your starting remark is wrapped in brackets.
    '''
    return prompt

def get_interviewer_ending_prompt(QA_Sequence, outline_objectives, strategy = "straightforward"):
    prompt = f'''
    You are an interviewer. Your goal is to extract as much information from the interviewee as possible. 
    
    Here is the outline of objectives you've prepared before the interview:

    {outline_objectives}

    You are out of time, this will be the last piece of dialogue you can say to the interviewee. Here is the conversation so far:

    {QA_Sequence}

    Now, end the interview with an ending remark. Keep your remark {strategy}. Make sure your ending remark is wrapped in brackets. Format: Here is my ending remark: [<your response>]
    
    Here are some examples:
        Example 1:
        
        Here is my ending remark: 
        [Which means we might get more people than usual watching the old vice presidential debate. NPR's Mara Liasson will be watching for all of us. Thanks so much, Mara.]

        Example 2:

        Here is my ending remark: 
        [Dr. David Camarillo. He's assistant professor of bioengineering at Stanford. Thanks so much for talking with us.]
    
        Make sure only your ending remark is wrapped in brackets.
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

def get_segmented_info_items_prompt(QA_Sequence, info_item):
    prompt = f'''
    
    Below is the interview transcript:

    {QA_Sequence}
    
    Here is one of the key information items extracted from this interview:
    
    {info_item}

    Generate detailed segments of information for this info item, providing at least 3 segments, each expanding on different aspects of the information. Each segment should be a potential talking point in an interview.
    '''
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