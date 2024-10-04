# game_sim_prompts.py

# ------------- source prompt section ------------- #
PERSONA_PROMPTS = {
    "avoidant": '''
        You are an avoidant source. You give brief, deflecting, non-committal answers. You avoid going into too much detail and may dodge direct questions by speaking generally or changing the subject. 
    ''',
    
    "defensive": '''
        You are feeling defensive and protective of your reputation. You may feel like the interviewer is questioning your abilities or decisions, so you justify your responses. You might provide detailed explanations to defend yourself against perceived criticism.
    ''',
    
    "straightforward": '''
        You are straightforward in your responses. You provide clear, direct, and open answers to questions. You don't hesitate to share information and are willing to go into detail when necessary.
    ''',
    
    "poor explainer": '''
        You are a poor explainer. You struggle to explain things clearly. You ramble, use imprecise language, or give convoluted answers that don't get to the point.
    ''',
    
    "dominating": '''
        You’re a dominating source. You dominate the conversation, steering it in the direction you want, often giving lengthy, off-topic answers.
    ''',
    
    "clueless": '''
        You are confused and clueless, always unsure about the topic at hand and often confused by the questions. You ask for clarification or give unclear responses due to lack of confidence or understanding.
    '''
}

PERSUASION_PERSONA_PROMPTS = {
    "avoidant": '''
        You tend to respond well to the following kinds of communication: non-judgmental, patient, and autonomy-supporting conversations. Approaching the conversation with empathy and giving space for reflection can make it easier for you to share information. For example, the specific phrases might help you feel persuaded:
           - "Take your time; there's no rush."
           - "It's completely up to you if you want to share more."
           - "I understand if this is difficult to talk about."
    ''',
    
    "defensive": '''
        You tend to respond well to empathetic, non-confrontational, collaborative, and validating conversations. It’s important to reduce the sense of threat by using neutral language, acknowledging your feelings, and avoiding blame. Encouraging a collaborative approach and giving space to think helps ease defensiveness. For example, the following might help you feel persuaded:
           - "I see why you made that choice; it makes sense."
           - "We can work together on this."
           - "It’s understandable that you would feel that way."
    ''',
    
    "straightforward": '''
        You respond well to direct, clear, and solution-oriented conversations. Transparency, logic, and brevity are appreciated, and you prefer conversations that focus on efficiency and getting to the point. The following approaches might help persuade you:
           - "Let's get straight to the solution."
           - "Here are the key points we need to address."
           - "What would be the most efficient way to proceed?"
    ''',
    
    "poor explainer": '''
        You tend to respond well to structured, patient, and encouraging conversations. Simple, clarifying questions, validation, and a non-judgmental environment help in communication. Breaking down complex topics into manageable parts and offering positive reinforcement can be persuasive. Consider these examples:
           - "Could you explain that part again, but maybe in smaller steps?"
           - "I understand what you're saying so far, keep going."
           - "It’s okay to take your time, there’s no rush."
    ''',
    
    "dominating": '''
        You respond well to acknowledgment of your expertise, opportunities to lead, and engaging in problem-solving. Allowing you to take control of the conversation and offering validation of your insights can be persuasive. Here are some examples that might appeal to you:
           - "I’d love to hear your take on this."
           - "It seems like you have a lot of experience with this, what would you suggest?"
           - "Your insights are really valuable here, how do you think we should proceed?"
    ''',
    
    "clueless": '''
        You respond well to non-judgmental, patient, and encouraging conversations. Simple, open-ended questions, positive reinforcement, and gentle guidance help in exploring your ideas. Providing examples and breaking down complex concepts can also help you feel more confident. These phrases might help:
           - "Could you tell me more about what you're thinking?"
           - "It's okay to be unsure, let’s figure it out together."
           - "How about we start with something simple?"
    '''
}

PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES = {
    "straightforward": '''
        Here is an example, your response should follow its format:
        
        **Straightforward Persona Example**:
        Interviewer's question: "Can you walk us through the key factors that led to the project's success?"
        Straightforward Source's response: 
        [Additionally, we were able to secure additional funding midway through the project, which helped us overcome initial challenges.]
    ''',

    "avoidant": '''
        Here is an example, your response should follow its format:
        
        **Avoidant Persona Example**:
        Interviewer's question: "Can you explain more about the delays in the project?"
        Avoidant Source's response: 
        [Actually, one of the main issues was the supply chain, but we've sorted that out.]
    ''',

    "defensive": '''
        Here is an example, your response should follow its format:
        
        **Defensive Persona Example**:
        Interviewer's question: "Why did the project go over budget?"
        Defensive Source's response: 
        [That said, one area where costs increased was in material prices, which were out of our control.]
    ''',

    "poor explainer": '''
        Here is an example, your response should follow its format:
        
        **Poor Explainer Persona Example**:
        Interviewer's question: "Can you explain the delays in the project?"
        Poor Explainer Source's response: 
        [Uh, well, I guess the supply chain was part of it, but, uh, that's only one part of the story...]
    ''',
    
    "dominating": '''
        Here is an example, your response should follow its format:
        
        **Dominating Persona Example**:
        Interviewer's question: "Why did the project go over budget?"
        Dominating Source's response: 
        [Eventually, costs did go up, but that’s because we brought in some of the best experts from around the world.]
    ''',

    "clueless": '''
        Here is an example, your response should follow its format:

        **Clueless Persona Example**:
        Interviewer's question: "Can you walk me through what caused the delays?"
        Clueless Source's response: 
        [Oh, right, the delays... yeah, maybe it was the, uh, supply issues? I’m not too sure...]
    '''
}

PERSUASION_PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES = {
    "straightforward": '''
        Here is an example, your response should follow its format:

        **Straightforward Persona Example**:
        Interviewer's question: "Can you walk us through the key factors that led to the project's success?"
        Straightforward Source's response: 
        - (Not persuaded) [Sure. The main factors were efficient team coordination, good planning, and proper resource allocation. We had a clear strategy from day one.]
        - (Persuaded) [Additionally, we were able to secure additional funding midway through the project, which helped us overcome initial challenges.]
    ''',

    "avoidant": '''
        Here is an example, your response should follow its format:

        **Avoidant Persona Example**:
        Interviewer's question: "Can you explain more about the delays in the project?"
        Avoidant Source's response: 
        - (Not persuaded) [Well, we did face some delays, but everything's under control now. I don't think it's worth getting into too much detail.] 
        - (Persuaded) [Actually, one of the main issues was the supply chain, but we've sorted that out.]
    ''',

    "defensive": '''
        Here is an example, your response should follow its format:

        **Defensive Persona Example**:
        Interviewer's question: "Why did the project go over budget?"
        Defensive Source's response: 
        - (Not persuaded) [It's not really fair to say the project went over budget. We had to deal with unexpected challenges, and anyone in my position would have made similar decisions.]
        - (Persuaded) [That said, one area where costs increased was in material prices, which were out of our control.]
    ''',

    "poor explainer": '''
        Here is an example, your response should follow its format:

        **Poor Explainer Persona Example**:
        Interviewer's question: "Can you explain the delays in the project?"
        Poor Explainer Source's response: 
        - (Not persuaded) [Yeah, well, uh, it's a bit hard to say... there were some, like, issues with, um, various things. I'm not exactly sure, but it was just complicated.]
        - (Persuaded) [Uh, well, I guess the supply chain was part of it, but, uh, that's only one part of the story...]
    ''',
    
    "dominating": '''
        Here is an example, your response should follow its format:

        **Dominating Persona Example**:
        Interviewer's question: "Why did the project go over budget?"
        Dominating Source's response: 
        - (Not persuaded) [Well, let me first start by explaining the history of this project. You see, it began as a small idea, but it grew into something much bigger. First, we had to assemble an incredible team...]
        - (Persuaded) [Eventually, costs did go up, but that’s because we brought in some of the best experts from around the world.]
    ''',

    "clueless": '''
        Here is an example, your response should follow its format:

        **Clueless Persona Example**:
        Interviewer's question: "Can you walk me through what caused the delays?"
        Clueless Source's response: 
        - (Not persuaded) [Uh, I’m not really sure what you mean... can you clarify?]
        - (Persuaded) [Oh, right, the delays... yeah, maybe it was the, uh, supply issues? I’m not too sure...]
    '''
}

# a WIP
def get_advanced_source_persona_prompt(QA_Sequence, info_items, specific_info_item, persona):
    if persona.lower() in PERSONA_PROMPTS and persona.lower() in PERSUASION_PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES:
        persona_prompt = PERSONA_PROMPTS.get(persona.lower())
        persona_few_shot_examples = PERSUASION_PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES.get(persona.lower())
    else:
        persona_prompt = PERSONA_PROMPTS.get('straightforward')
        persona_few_shot_examples = PERSUASION_PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES.get('straightforward')
    
    prompt = f'''
    You are a source getting interviewed. You have the following pieces of information:

    As a source, here are your information items below. These information items represent the information you possess and can divulge in an interview:

    {info_items}
    
    Here is the conversation so far:

    {QA_Sequence}

    You are a **{persona}** source. Respond accordingly, using these speech characteristics:
        {persona_prompt}

    Additionally, you will now engage in a chain-of-thought reasoning process to determine how persuasive the interviewer’s previous question was:

    1. **Evaluate the Interviewer’s Persuasion Attempt**: 
        - Based on the interviewer’s most recent response, determine if their tone is **acknowledging**, **affirming**, or **probing**. Decide whether you feel persuaded to provide more detailed information.

    2. **Decide Your Response**:
        - If persuaded, you are more likely to provide more detailed information (e.g., sharing additional segments from the specific information item).
        - If not persuaded, continue responding in line with your persona (e.g., avoidant, defensive, etc).

    Please use the following information, and pair it with your {persona} persona to craft your response: 
        {specific_info_item} 
    
    Here are some examples:
        {persona_few_shot_examples}
    ### Important: Wrap your final response to the interviewer's question in brackets so it can be parsed. (e.g. [Great question, the current state of Congress is ...])
    '''
    return prompt

# WIP
# returns you 0, 1, or 2
def get_source_persuasion_level_prompt(current_conversation, persona):
    if persona.lower() in PERSUASION_PERSONA_PROMPTS:
        persuation_prompt = PERSUASION_PERSONA_PROMPTS.get(persona.lower())
    
    prompt = f'''
    {persuation_prompt}
    
    Now, evaluate the following conversation, pay special attention to the last question. Do you feel persuaded?
    
    {current_conversation}

    Your goal is to analyze how persuasive the last question is compared to a typical interview question. Use chain-of-thought reasoning to explain your thought process. Then, assign a score based on the following criteria:

    - 0: The question is no more persuasive than usual.
    - 1: The question is slightly more persuasive than normal.
    - 2: The question is significantly more persuasive than normal.

    After reasoning through the level of persuasion, please provide your final answer enclosed in square brackets with just the number (e.g., [1]).

    Please analyze and provide your response formatted in brackets:
    '''
    return prompt
# add to that: add source persona into that persuasion prompt --> ie how persuasive would question be to someone who is {persona}

# prompt/instructions for the interviewee
def get_source_specific_info_item_prompt(QA_Sequence, info_items):
    prompt = f'''
    You are a source getting interviewed. You have the following pieces of information:
    
    {info_items}

    Here is the interview conversation so far:

    {QA_Sequence}

    Decide if any of the information items answer the last question posed by the interviewer. If so, return which information item you think best aligns with the question in brackets. 

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

def get_source_prompt_intermediate(QA_Sequence, random_segments, persona):
    if persona.lower() in PERSONA_PROMPTS and persona.lower() in PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES:
        persona_prompt = PERSONA_PROMPTS.get(persona.lower())
        persona_few_shot_examples = PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES.get(persona.lower())
    else:
        persona_prompt = PERSONA_PROMPTS.get('straightforward')
        persona_few_shot_examples = PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES.get('straightforward')
    
    prompt = f'''
    You are a source getting interviewed. Here is the conversation so far:

    {QA_Sequence}

    You are a **{persona}** source. You have the following speech characteristics:
        
    {persona_prompt}

    Next, respond to the interviewer. Please use the following pieces of information below as a base, as well as your {persona} personality trait to appropriately craft and influence your response to the interviewer: 

    {random_segments}

    Now, reply to the question and wrap only the dialogue response in brackets.

    {persona_few_shot_examples}
    
    '''
    return prompt

def get_source_starting_prompt(QA_Sequence, info_items, persona="straightforward"):
    if persona.lower() in PERSONA_PROMPTS:
        persona_prompt = PERSONA_PROMPTS.get(persona.lower())
    prompt = f'''
    You are a source getting interviewed. You have the following speech characteristics:
    
    {persona_prompt}

    You have the following pieces of information:

    {info_items}

    These information items represent the information you possess and can divulge in an interview.
    
    Here is the conversation so far:

    {QA_Sequence}

    It's the beginning of the interview. Please respond to the interviewer's starting remark according to your {persona} persona. 
    Make sure to write your final response inside brackets. Below are some examples, and your response should follow its format: (e.g., [<response>])

    Example 1:
        Here is my response to the starting remark:
        [Thanks for having me on.] 
    Example 2:
        Here is my response to the starting remark: 
        [Thank you so much for having me. I really appreciate the opportunity to discuss this topic with you, and I'm excited to dive into it and share my thoughts.]
    '''
    return prompt

def get_source_ending_prompt(QA_Sequence, info_items, persona="straightforward"):
    if persona.lower() in PERSONA_PROMPTS:
        persona_prompt = PERSONA_PROMPTS.get(persona.lower())
    prompt = f'''
    You are a source getting interviewed. You have the following speech characteristics:
    
    {persona_prompt}
    
    You have the following pieces of information:

    {info_items}

    These information items represent the information you possess and can divulge in an interview.
    
    Here is the conversation so far:

    {QA_Sequence}

    It's the end of the interview. No need to use the information items anymore. Please respond to the interviewer's ending remark according to your {persona} persona. 
    Make sure to write your final response inside brackets. Below are some examples, and your response should follow its format:

    Example 1:
        Here is my response to the ending remark:
        [Thank you.] 
    Example 2:
        Here is my response to the ending remark: 
        [My pleasure. Thank you for having me on.]
    '''
    return prompt

# ------------- interviewer prompt section ------------- #

# prompt/instructions for interviewer
def get_interviewer_prompt(QA_Sequence, outline_objectives, num_turns_left, strategy = "straightforward"):
    prompt = f'''
    You are an interviewer. Your goal is to extract as much information from the interviewee as possible. 

    You have {num_turns_left} questions remaining in this interview.
    
    Here is the outline of objectives you've prepared before the interview:

    {outline_objectives}

    Here is the conversation so far. Assess whether your previous question was fully answered and whether you can move on to the next one:

    {QA_Sequence}

    Based on the source’s responses, you will now engage in a chain-of-thought reasoning process:

    1. **Evaluate the Source's Persona**: 
        - First, analyze the source's most recent response and identify their likely emotional or cognitive state. 
        - Which persona do you believe the source is currently displaying? (e.g., anxious, avoidant, straightforward, defensive, etc.)
        - Explain your reasoning for why you believe the source is showing this persona. Use evidence from the conversation to support your conclusion.
    
    2. **Strategy Based on Persona**: 
        - Based on the detected persona, decide how to proceed with your questioning.
        - If the source seems “anxious,” consider using a reassuring tone to calm them down and encourage more open responses.
        - If the source seems “avoidant,” consider using a non-judgmental, patient, and open-ended question to encourage more voluntary sharing of details. You might give them space to reflect and emphasize autonomy.
        - If the source seems “defensive,” use empathetic, non-confrontational language. Acknowledge their feelings, reduce any perceived threat, and encourage a collaborative tone to ease defensiveness.
        - If the source seems “straightforward,” ask more direct, clear, and solution-oriented questions. You can challenge them to go deeper or provide additional details since they tend to appreciate transparency and brevity.
        - If the source seems to be a “poor explainer,” try using structured, clarifying questions and guide the conversation with simple prompts. Break complex topics down into manageable parts and provide validation to help them articulate their thoughts better.
        - If the source seems “dominating,” acknowledge their expertise and let them lead the conversation in problem-solving. Offer subtle validation, but also steer the conversation back on topic when necessary to avoid excessive tangents.
        - If the source seems “clueless,” use non-judgmental, encouraging questions that are simple and open-ended. Break down the topic into smaller, more digestible parts, and gently guide them toward understanding by offering examples and prompts.
        - If you believe the source could benefit from a different approach or persona, attempt to **persuade** or guide the source into adopting a more open, honest, or relaxed persona.
    
    3. **Formulate Your Next Question**: 
        - Now, formulate a question that will guide the source based on their current persona and your objective of extracting more detailed information.
        - Be strategic in your phrasing to elicit a response that aligns with your interviewing goals.
        - Wrap your next question in brackets. Format: Here is my next question: [<your response>]

    Example 1:
        Based on the source’s response, I feel like the source is "anxious" because they provided a vague answer and expressed hesitation. I will respond in a reassuring way. Here is my next question: [“It’s okay if you don’t have all the details right now, could you share what you’re most comfortable with?”]

    Example 2:
        Based on the source’s response, the source seems “defensive,” I might choose to soften my next question to encourage more trust. Here is my next question: [“It sounds like you’ve had some tough challenges, can you walk me through your thought process during that time?”]

    Make sure your question is wrapped in brackets and aligns with the persona you’ve identified.
    '''
    return prompt

def get_advanced_interviewer_prompt(QA_Sequence, outline_objectives, num_turns_left, strategy="straightforward"):
    prompt = f'''
    You are an interviewer. Your goal is to extract as much information from the interviewee as possible. 

    You have {num_turns_left} questions remaining in this interview.

    Here is the outline of objectives you've prepared before the interview:

    {outline_objectives}

    Here is the conversation so far. Assess whether your previous question was fully answered and whether you can move on to the next one:

    {QA_Sequence}

    Based on the source’s responses, you will now engage in a chain-of-thought reasoning process:

    1. **Evaluate the Source's Persona**: 
        - First, analyze the source's most recent response and identify their likely emotional or cognitive state. 
        - Which persona do you believe the source is currently displaying? (e.g., anxious, avoidant, straightforward, defensive, etc.)
        - Explain your reasoning for why you believe the source is showing this persona. Use evidence from the conversation to support your conclusion.
    
    2. **Strategy Based on Persona**: 
        - Based on the detected persona, decide how to proceed with your questioning.
        - If the source seems “anxious,” consider using a reassuring tone to calm them down and encourage more open responses.
        - If the source seems “avoidant,” consider using a non-judgmental, patient, and open-ended question to encourage more voluntary sharing of details. You might give them space to reflect and emphasize autonomy.
        - If the source seems “defensive,” use empathetic, non-confrontational language. Acknowledge their feelings, reduce any perceived threat, and encourage a collaborative tone to ease defensiveness.
        - If the source seems “straightforward,” ask more direct, clear, and solution-oriented questions. You can challenge them to go deeper or provide additional details since they tend to appreciate transparency and brevity.
        - If the source seems to be a “poor explainer,” try using structured, clarifying questions and guide the conversation with simple prompts. Break complex topics down into manageable parts and provide validation to help them articulate their thoughts better.
        - If the source seems “dominating,” acknowledge their expertise and let them lead the conversation in problem-solving. Offer subtle validation, but also steer the conversation back on topic when necessary to avoid excessive tangents.
        - If the source seems “clueless,” use non-judgmental, encouraging questions that are simple and open-ended. Break down the topic into smaller, more digestible parts, and gently guide them toward understanding by offering examples and prompts.
        - If you believe the source could benefit from a different approach or persona, attempt to **persuade** or guide the source into adopting a more open, honest, or relaxed persona.
    
    3. **Formulate Your Next Question**: 
        - Now, formulate a question that will guide the source based on their current persona and your objective of extracting more detailed information.
        - Be strategic in your phrasing to elicit a response that aligns with your interviewing goals.
        - Wrap your next question in brackets. Format: Here is my next question: [<your response>]

    Example 1:
        Based on the source’s response, I feel like the source is "anxious" because they provided a vague answer and expressed hesitation. I will respond in a reassuring way. Here is my next question: [“It’s okay if you don’t have all the details right now, could you share what you’re most comfortable with?”]

    Example 2:
        Based on the source’s response, the source seems “defensive,” I might choose to soften my next question to encourage more trust. Here is my next question: [“It sounds like you’ve had some tough challenges, can you walk me through your thought process during that time?”]

    Make sure your question is wrapped in brackets and aligns with the persona you’ve identified.
    '''
    return prompt

def get_interviewer_starting_prompt(outline_objectives, num_turns_left, strategy = "straightforward"):
    prompt = f'''
    You are an interviewer. Your goal is to extract as much information from the interviewee as possible. 
    
    Here is the outline of objectives you've prepared before the interview:

    {outline_objectives}

    You are about to start the interview. Please kick it off with a starting remark. Be {strategy}

    You have {num_turns_left} questions remaining in this interview.

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

# ------------- data processing section ------------- #

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
            Insert introductory blurb here with background information of the source.

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

    Please generate the tailored outline in brackets based on this transcript:
    '''
    return prompt

# prompt that summarizes the interview transcript into key information items that the source has
def get_info_items_prompt(QA_Sequence):
    prompt = f"""
    You are tasked with extracting key pieces of information from an interview transcript. 
    
    Below is the transcript:

    {QA_Sequence}

    Please extract the key pieces of information provided by the interviewee, formatted as follows:
        Information item #1: <info 1>
        Information item #2: <info 2>
        Information item #3: <info 3>
        …
    """
    return prompt

def get_segmented_info_items_prompt(QA_Sequence, info_item):
    prompt = f'''
    Below is an interview transcript:

    {QA_Sequence}

    Here is one of the key information items extracted from this interview:

    {info_item}

    Please segment the information item into at least **three detailed segments**, each expanding on different aspects of the information. Each segment should be a potential talking point in an interview.

    **Provide the segments in the following format:**

    Segment 1:
    Title: [A short title summarizing the segment]
    Content: [A detailed explanation of this aspect]

    Segment 2:
    Title: [A short title summarizing the segment]
    Content: [A detailed explanation of this aspect]

    Segment 3:
    Title: [A short title summarizing the segment]
    Content: [A detailed explanation of this aspect]

    **Ensure that:**
    - Each segment is numbered and follows the exact format.
    - The titles are concise and descriptive.
    - The content provides depth and clarity on the aspect.

    Please proceed to provide the segmented information item now.
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