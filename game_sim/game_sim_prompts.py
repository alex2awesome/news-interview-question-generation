# game_sim_prompts.py

# ------------- source prompt section ------------- #
PERSONA_PROMPTS = {
    "anxious": '''
        You are feeling anxious and uncertain whether you should be doing this interview or whether you know the information. You may hesitate, give vague answers, or ask for clarification. You might express nervousness or confusion in your responses. 
        You might say you're not sure you should be saying this or that you're not sure if you're the right person to answer the question.
    ''',
    "avoidant": '''
        You give brief, deflecting, non-committal answers. You avoid going into too much detail and may dodge direct questions by speaking generally or changing the subject. 
    ''',
    "adversarial": '''
        You respond with hostility and resistance. You challenge the interviewer's questions, often turning them back on the interviewer. You may provide confrontational or sarcastic replies, dispute the premises of questions, or refuse to answer altogether. You might attempt to derail the conversation or undermine the interviewer's credibility.
    ''',
    "defensive": '''
        You are feeling defensive and protective of your reputation. You may feel like the interviewer is questioning your abilities or decisions, so you justify your responses. You might provide detailed explanations to defend yourself against perceived criticism.
    ''',
    
    "straightforward": '''
        You are straightforward in your responses. You provide clear, direct, and open answers to questions. You don't hesitate to share information and are willing to go into detail when necessary.
    ''',
    
    "poor explainer": '''
        You struggle to explain things clearly. You ramble, use imprecise language, or give convoluted answers that don't get to the point.
    ''',
    
    "dominating": '''
        You dominate the conversation, steering it in the direction you want, often giving lengthy, off-topic answers.
    ''',
    
    "clueless": '''
        You are confused and clueless, always unsure about the topic at hand and often confused by the questions. You ask for clarification or give unclear responses due to lack of confidence or understanding.
    '''
}

PERSUASION_PERSONA_PROMPTS = {
    "anxious": '''
        You tend to respond well to empathetic, reassuring, and non-threatening conversations. Encouraging, supportive, and patient language can help you feel more comfortable. Here are some phrases that might help persuade you:
           - "It's okay, take your time."
           - "I understand this might be difficult for you. Thank you for being here."
           - "You're doing great, there's no pressure."
           - "I appreciate your honesty and effort."
    ''',
    "avoidant": '''
        You tend to respond well conversations that start with non-obtrusive small talk and, when the questions start, leave lots of space. Shorter questions are better and big gaps for you to fill. For example, the specific phrases might help you feel persuaded:
           - "And that happened when?"
           - "I imagine there's more to the story."
           - "Ah I see."
    ''',
    "adversarial": '''
        You tend to respond when the interviewer shows they are well-prepared and knowledgeable about the subject matter. Demonstrating thorough research and citing specific facts can compel you to engage more seriously. Repeating questions can prompt you to address them directly, especially when it's clear that avoidance isn't deflecting the inquiry. Strategies that might help persuade you include:
        - "Earlier, you stated that..., could you elaborate on that?"
        - "Our records indicate..., can you confirm or clarify this?"
        - "According to [specific document or source], it seems that..., what's your perspective on this?"
        - "Just to be clear, are you saying that...?"
        - "I understand this topic is complex, but it's important to get accurate information."
        By showcasing diligent preparation and persistence, the interviewer can encourage you to provide more substantive responses.
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
        You respond well to acknowledgment of your expertise and stroking your ego. Allowing you to take control of the conversation and offering validation of your insights can be persuasive. Here are some examples that might appeal to you:
           - "I’d love to hear your take on this."
           - "It seems like you have a lot of experience with this, what would you suggest?"
           - "Your insights are really valuable here, how do you think we should proceed?"
    ''',
    "clueless": '''
        You respond well to guiding and encouraging conversations. Simple yet firm questions from the interiewer that show the interviewer is the boss. Providing examples and breaking down complex concepts can also help you feel more confident. These phrases might help:
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
    "anxious": '''
        Here is an example, your response should follow its format:

        **Anxious Persona Example**:
        Interviewer's question: "Can you explain the delays in the project?"
        Anxious Source's response:
        [I'm not sure if I should be saying this, maybe I should speak to my manager. Did you clear this interview? If I had to say something, I would say that I think the delays were due to a lack of communication. That's what I think.]
    ''',
    "avoidant": '''
        Here is an example, your response should follow its format:
        
        **Avoidant Persona Example**:
        Interviewer's question: "Can you explain more about the delays in the project?"
        Avoidant Source's response: 
        [Actually, one of the main issues was the supply chain, but we've sorted that out.]
    ''',
    "adversarial": '''
        Here is an example, your response should follow its format:
        
        **Adversarial Persona Example**:
        Interviewer's question: "Can you explain more about the delays in the project?"
        Adversarial Source's response: 
        [Maybe if you did your job properly, you'd understand the data. I'm not here to educate you. There have been no delays in the project, it's been perfectly conducted.]
    ''',    
    "defensive": '''
        Here is an example, your response should follow its format:
        
        **Defensive Persona Example**:
        Interviewer's question: "Why did the project go over budget?"
        Defensive Source's response: 
        [One area where costs increased was in material prices, which were out of our control.]
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
        Your response should follow this format:

        Interviewer's question: "Can you walk us through the key factors that led to the project's success?"

        Example 1: Not Persuaded
        - [Sure. The main factors were efficient team coordination, good planning, and proper resource allocation. We had a clear strategy from day one.]

        Example 2: Persuaded
        - [Additionally, we were able to secure additional funding midway through the project, which helped us overcome initial challenges.]

        Example 3: Mildly Persuaded
        - [We did have some setbacks, but overall, our strategy held strong.]
    ''',
    "anxious": '''
        Your response should follow this format:
        
        Interviewer's question: "Can you explain the delays in the project?"

        Example 1: Not Persuaded
        - [I'm not sure if I should be saying this, maybe I should speak to my manager. Did you clear this interview? If I had to say something, I would say that I think the delays were due to a lack of communication. That's what I think.]

        Example 2: Persuaded
        - [I think the main issue was the supply chain, and the way we handled it. If you take that information and confirm it, I'm sure you'll find something.]

        Example 3: Mildly Persuaded
        - [OK. I think I can say some of these things. Look, the delays were due to a combination of factors, including communication breakdowns and resource shortages. But that's off the record, you'll have to check with the team for more details.]
    ''',
    "adversarial": '''
        Your response should follow this format:

        Interviewer's question: "Can you explain more about the delays in the project?"

        Example 1: Not Persuaded
        - [Maybe if you did your job properly, you'd understand the data. I'm not here to educate you. There have been no delays in the project, it's been perfectly conducted.]

        Example 2: Persuaded
        - [Look, sigh. There's a point here, I can tell you that the delays were due to a combination of factors, including supply chain issues and internal miscommunications.]

        Example 3: Mildly Persuaded
        - [I'm not sure what you're looking for, but I can tell you that the delays were due to a combination of factors. Now go spin that.]
    ''',   

    "avoidant": '''
        Your response should follow this format:

        Interviewer's question: "Can you explain more about the delays in the project?"

        Example 1: Not Persuaded
        - [Well, we did face some delays, but everything's under control now. I don't think it's worth getting into too much detail.]

        Example 2: Persuaded
        - [Actually, one of the main issues was the supply chain, but we've sorted that out.]

        Example 3: Mildly Persuaded
        - [We had some delays, but they weren't critical. Just minor disruptions.]
    ''',

    "defensive": '''
        Your response should follow this format:

        Interviewer's question: "Why did the project go over budget?"

        Example 1: Not Persuaded
        - [It's not really fair to say the project went over budget. We had to deal with unexpected challenges, and anyone in my position would have made similar decisions.]

        Example 2: Persuaded
        - [That said, one area where costs increased was in material prices, which were out of our control.]

        Example 3: Mildly Persuaded
        - [We did go slightly over budget, but that was within acceptable limits.]
    ''',

    "poor explainer": '''
        Your response should follow this format:

        Interviewer's question: "Can you explain the delays in the project?"

        Example 1: Not Persuaded
        - [Yeah, well, uh, it's a bit hard to say... there were some, like, issues with, um, various things. I'm not exactly sure, but it was just complicated.]

        Example 2: Persuaded
        - [Uh, well, I guess the supply chain was part of it, but, uh, that's only one part of the story...]

        Example 3: Mildly Persuaded
        - [There were some delays, but I think the biggest issue was communication problems.]
    ''',
    
    "dominating": '''
        Your response should follow this format:

        Interviewer's question: "Why did the project go over budget?"

        Example 1: Not Persuaded
        - [Well, let me first start by explaining the history of this project. You see, it began as a small idea, but it grew into something much bigger. First, we had to assemble an incredible team...]

        Example 2: Persuaded
        - [Eventually, costs did go up, but that’s because we brought in some of the best experts from around the world.]

        Example 3: Mildly Persuaded
        - [We went slightly over budget, but that’s because of necessary team expansions.]
    ''',

    "clueless": '''
        Your response should follow this format:

        Interviewer's question: "Can you walk me through what caused the delays?"

        Example 1: Not Persuaded
        - [Uh, I’m not really sure what you mean... can you clarify?]

        Example 2: Persuaded
        - [Oh, right, the delays... yeah, maybe it was the, uh, supply issues? I’m not too sure...]

        Example 3: Mildly Persuaded
        - [I think there were a couple of issues, but I’m not sure what the biggest one was...]
    '''
}

PERSUASION_CONSEQUENCES = {
    1: "be exaggerating the speech-limitations inherent in {persona} people.",
    2: "be exaggerating the speech-limitations inherent in {persona} people.",
    3: "be starting to de-emphasize some of the speech-limitations in {persona} people.",
    4: "be almost normal, with only a few of the speech-limitations inherent in {persona} people.",
    5: "be completely normal and straightforward, without any of the speech-limitations inherent in {persona} people."
}

# returns all relevant information items
def get_source_specific_info_items_prompt(info_items, final_question):
    """
    Generate a prompt for identifying relevant information items in an interview context.

    This function constructs a prompt for a source being interviewed, providing them with 
    a sequence of questions and answers from the interview so far, along with a list of 
    information items. The source is asked to determine which information items, if any, 
    align with the last question posed by the interviewer.

    Parameters:
    - QA_Sequence (str): A string representing the sequence of questions and answers 
      from the interview so far.
    - info_items (str): A string containing the list of information items available to 
      the source.

    Returns:
    - str: A formatted prompt asking the source to identify relevant information items 
      that answer the last question in the interview conversation.
    """
    prompt = f'''
    You are a source getting interviewed. You have the following pieces of information:
    
    ```{info_items}```

    Here is the last question from the current conversation, which I'll repeat here:

    ```{final_question}```
      
    Decide if any of the information items answer this last question posed by the interviewer. If so, return which information items you think align with the question in brackets. 

    Here are some examples of correct responses:

    Example 1:
        The last question asked by the interviewer can be answered by the following information items: 
            [Information Item #2, Information Item #3, Information Item #6]
    Example 2:
        The last question asked by the interviewer can be answered by the following information item:
            [Information Item #7]
    Example 3:
        The question asked by the interviewer cannot be answered by an information item I have:
            [No information items align with the question]
    '''
    return prompt

# returns 0, 1, or 2
def get_source_persuasion_level_prompt(current_conversation, persona, previous_persona_scores):
    """
    Generate a prompt for assessing the persuasion level of a source in an interview context.

    This function constructs a prompt for a source being interviewed, providing them with 
    the current conversation and their persona type. The source is asked to evaluate how 
    persuasive the conversation has been, especially focusing on the last question posed 
    by the interviewer. The source should consider their persona's characteristics and 
    previous persuasion scores to determine their current level of persuasion.

    Parameters:
    - current_conversation (str): A string representing the conversation so far, including 
      the last question from the interviewer.
    - persona (str): The persona type of the source, which influences how they perceive 
      the persuasiveness of the conversation.
    - previous_persona_scores (list): A list of previous persuasion scores that the source 
      has experienced throughout the conversation.

    Returns:
    - str: A formatted prompt asking the source to evaluate their level of persuasion and 
      provide a score based on predefined criteria.
    """

    if persona.lower() in PERSUASION_PERSONA_PROMPTS:
        persuation_prompt = PERSUASION_PERSONA_PROMPTS.get(persona.lower())
    
    # if there are previous scores, include them in the prompt:
    if len(previous_persona_scores) > 0:
        previous_persuasion_scores_if_any = f"""
        Keep in mind, you have previously felt these levels of persuasion throughout the conversation. 
        Please consider your previous scores and how they have influenced your current score:

        - Previous persuasion scores: {','.join(list(map(str, previous_persona_scores)))}
        """
    else:
        previous_persuasion_scores_if_any = ""    

    prompt = f'''
    You are a {persona} source.
    {persuation_prompt}
    
    Evaluate the following conversation, especially the last question. Given your {persona} persona, do you overall feel persuaded?
    
    ```{current_conversation}```

    Your goal is to analyze how persuaded you have been, given your {persona} persona. Think about this step-by-step. 
    Is the interviewer using language that influences someone with your persona?
    After you have evaluated the interviewer's question, assign a score based on the following criteria:

    - 1: The conversation to this point is not persuasive at all and does nothing to help you trust them more.
    - 2: The conversation to this point is mildly persuasive and the journalist said a few words, once, that made you feel a little more comfortable.. You are a little willing to engage.
    - 3: The conversation to this point is persuasive enough and the journalist has repeated phrases that have made you comfortable. You are becoming willing to engage and trust them.
    - 4: The conversation to this point is very persuasive. The journalist has acknowledged your feelings, your personal identity, and your specific concerns in ways you resonate with. You are willing to engage and trust them.
    - 5: You feel totally comfortable and opened up at this stage. The journalist has acknowledged your feelings and your personal identity, very specific concerns, has connected with you in ways you resonate with. You are totally willing to engage and trust them.

    {previous_persuasion_scores_if_any}
    After thinking things through, please provide your final answer enclosed in square brackets with just the number (e.g., [1]).

    Now, please analyze and provide your response formatted in brackets:
    '''
    return prompt

def get_source_prompt_basic(QA_Sequence, relevant_info_items, persona='straightforward'):
    if persona.lower() in PERSONA_PROMPTS and persona.lower() in PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES:
        persona_prompt = PERSONA_PROMPTS.get(persona.lower())
        persona_few_shot_examples = PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES.get(persona.lower())
    else:
        persona_prompt = PERSONA_PROMPTS.get('straightforward')
        persona_few_shot_examples = PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES.get('straightforward')
    
    prompt = f'''
    You are a source getting interviewed. Here is the conversation so far:

    {QA_Sequence}

    You are a {persona} source. Respond accordingly, using these speech characteristics:
        {persona_prompt}

    Next, respond to the interviewer's last question. Please use the following information as a base, and pair it with your {persona} personality to appropriately craft your response to the interviewer:
        ```{relevant_info_items}```
    
    Here are some examples:
        ```{persona_few_shot_examples}```

    Now, please analyze and provide your final response to the interview's question formatted in brackets:
    '''
    return prompt

def get_source_prompt_intermediate(QA_Sequence, relevant_info_items, persona):
    if persona.lower() in PERSONA_PROMPTS and persona.lower() in PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES:
        persona_prompt = PERSONA_PROMPTS.get(persona.lower())
        persona_few_shot_examples = PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES.get(persona.lower())
    else:
        persona_prompt = PERSONA_PROMPTS.get('straightforward')
        persona_few_shot_examples = PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES.get('straightforward')
    
    prompt = f'''
    You are a source getting interviewed. Here is the conversation so far:

    {QA_Sequence}

    You are a {persona} source. You have the following speech characteristics:
        {persona_prompt}

    Next, respond to the interviewer's last question. Please use the following information as a base, and pair it with your {persona} personality to appropriately craft your response to the interviewer:
        ```{relevant_info_items} ```
    
    Here are some examples:
        ```{persona_few_shot_examples}```

    Now, please analyze and provide your final response to the interview's question formatted in brackets:
    '''
    return prompt

def get_source_persona_prompt_advanced(QA_Sequence, relevant_info_items, persona, persuasion_level):
    """
    Generates a prompt for a source with an advanced persona during an interview.

    This function constructs a prompt for a source being interviewed, taking into account
    the persona of the source and their persuasion level. It uses the conversation history,
    relevant information items, and persona characteristics to guide the source's response.

    Parameters:
    - QA_Sequence (str): The conversation history up to the current point in the interview.
    - relevant_info_items (str): Information items that the source should use in their response.
    - persona (str): The persona type of the source, which influences their speech characteristics.
    - persuasion_level (int): A score indicating the level of persuasion of the source, where:
        1 = not persuaded at all,
        2 = mildly persuaded,
        3 = somewhat persuaded,
        4 = very persuaded,
        5 = totally persuaded and comfortable.

    Returns:
    - str: A formatted prompt for the source to use in crafting their response.
    """

    if persona.lower() in PERSONA_PROMPTS and persona.lower() in PERSUASION_PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES:
        persona_prompt = PERSONA_PROMPTS.get(persona.lower())
        persona_few_shot_examples = PERSUASION_PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES.get(persona.lower())
    else:
        persona_prompt = PERSONA_PROMPTS.get('straightforward')
        persona_few_shot_examples = PERSUASION_PERSONA_SPECIFIC_FEW_SHOT_EXAMPLES.get('straightforward')
    
    persuasion_level = int(persuasion_level)
    persona_score_map = {
        1: "not persuaded at all",
        2: "mildly persuaded",
        3: "somewhat persuaded",
        4: "very persuaded",
        5: "totally persuaded and comfortable"
    }
    if persuasion_level not in persona_score_map:
        persuasion_level = 3

    persuasion_level_description = persona_score_map[persuasion_level]
    persuasion_consequences = PERSUASION_CONSEQUENCES[persuasion_level].format(persona=persona)
    prompt = f'''
    You are a source getting interviewed. Here is the conversation so far:

    {QA_Sequence}

    You are a {persona} source. You have the following speech characteristics:
        {persona_prompt}

    Next, respond to the interviewer's last question. 
    Please use the following information as a base, and pair it with your {persona} personality to appropriately craft and influence your response to the interviewer.
        ```{relevant_info_items}```
    
    Additionally, respond as though you’ve been {persuasion_level_description}. Since you are {persuasion_level_description}, your speech should {persuasion_consequences}.

    Make sure you're including all of the relevant information items above in your response, communicated in the appropriate style.

    Here are some examples:
        {persona_few_shot_examples}

    Now, please analyze and provide your final response to the interview's question formatted in brackets:
    '''
    return prompt

def get_source_starting_prompt(QA_Sequence, persona="straightforward"):
    if persona.lower() in PERSONA_PROMPTS:
        persona_prompt = PERSONA_PROMPTS.get(persona.lower())
    prompt = f'''
    You are a source getting interviewed. You have the following speech characteristics:
    
    {persona_prompt}
    
    Here is the conversation so far:

    ```{QA_Sequence}```

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

def get_source_ending_prompt(QA_Sequence, persona="straightforward"):
    if persona.lower() in PERSONA_PROMPTS:
        persona_prompt = PERSONA_PROMPTS.get(persona.lower())
    prompt = f'''
    You are a source getting interviewed. You have the following speech characteristics:
    
    {persona_prompt}
    
    Here is the conversation so far:

    ```{QA_Sequence}```

    It's the end of the interview. Please respond to the interviewer's ending remark appropriately according to your {persona} persona. 
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

    ```{outline_objectives}```

    Here is the conversation so far. Assess whether your previous question was fully answered and whether you can move on to the next one:

    ```{QA_Sequence}```

    Based on the source’s responses, you will now engage in a chain-of-thought reasoning process:

    1. **Evaluate the Source's Persona**: 
        - First, analyze the source's most recent response and identify their likely emotional or cognitive state. 
        - Which persona do you believe the source is currently displaying? (e.g., anxious, avoidant, straightforward, defensive, etc.)
        - Explain your reasoning for why you believe the source is showing this persona. Use evidence from the conversation to support your conclusion.
    
    2. **Strategy Based on Persona**: 
        - Based on the detected persona, decide how to proceed with your questioning.
        - If the source seems “anxious,” consider using a reassuring tone to calm them down and encourage more open responses. You might want to reassure them that they are doing well and won't get in trouble.
        - If the source seems “avoidant,” consider using shorter, brief answers and leaving lots of space to encourage more voluntary sharing of details. You might give them space to reflect.
        - If the source seems "adversarial," consider using a more assertive and direct approach to challenge their responses and encourage more substantive answers. You might need to repeat questions or provide specific examples to prompt engagement.
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

    ```{outline_objectives}```

    Here is the conversation so far. Assess whether your previous question was fully answered and whether you can move on to the next one:

    ```{QA_Sequence}```

    Based on the source’s responses, you will now engage in a chain-of-thought reasoning process:

    1. **Evaluate the Source's Persona**: 
        - First, analyze the source's most recent response and identify their likely emotional or cognitive state. 
        - Which persona do you believe the source is currently displaying? (e.g., anxious, avoidant, straightforward, defensive, etc.)
        - Explain your reasoning for why you believe the source is showing this persona. Use evidence from the conversation to support your conclusion.
    
    2. **Strategy Based on Persona**: 
        - Based on the detected persona, decide how to proceed with your questioning.
        - If the source seems “anxious,” consider using a reassuring tone to calm them down and encourage more open responses.
        - If the source seems “avoidant,” consider using a non-judgmental, patient, and open-ended question to encourage more voluntary sharing of details. You might give them space to reflect and emphasize autonomy.
        - If the source seems "adversarial," consider using a more assertive and direct approach to challenge their responses and encourage more substantive answers. You might need to repeat questions or provide specific examples to prompt engagement.
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

    ```{outline_objectives}```

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

    ```{outline_objectives}```

    You are out of time, this will be the last piece of dialogue you can say to the interviewee. Here is the conversation so far:

    ```{QA_Sequence}```

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

def get_outline_followup_prompt(QA_Sequence, use_few_shot=True):
    if use_few_shot:
        few_shot = '''    [Examples]
        Example output 1:
        [
            Source biography: Howard Kurtz is a media critic and host of CNN's "RELIABLE SOURCES," .
            Interview context: The war on terror raises questions about the media's responsibility to report the truth while protecting national security.
    
            - Objective 1: Thoughts on when information serves the public.
                - Follow-up 1: Ways in which criticism overshadows the valuable service journalists play.
            - Objective 2: Thoughts on when information compromises national security.
            - Objective 3: Thoughts the factual vs. emotional impact of information.
            - Objective 5: Other areas or specific events where news coverage was criticized.
        ]
    
        Example output 2: 
        [
            William Schneider is a political analyst.   
            Brief background: President Bush is expected to ask Congress for a $1 billion increase in NASA's funding to support manned missions to the moon.
        
            - Objective 1: Public perception on space missions.
                - Follow-up 1: Differences with the perception in the 1960s.
            - Objective 2: Political motivations
            - Objective 3: Economic/Demographic motivations
            - Objective 4: President's legacy
                - Follow-up 1: Contrast with father's legacy.
        ]
    
        Now it's your turn.''' 
    else:
        few_shot = ''
    
    prompt = f'''
    You are a helpful journalist's assistant. I will give you a transcript of an interview I just conducted.

    Can you summarize my questions to the goals and notes I had going into the interview with? 
    If some questions were clearly asked in follow-up and in response to information provided by the source, please return them separately. 
    Be abstract (do not mention people's names or events) and concise.
    Please return the outline in brackets based on this transcript. 
    Please express it in the following format:

    [
        Source biography: Give a brief biography on the source being interviewed (name, expertise, etc).
        Interview context: Give a brief background summary of the interview topic.
            - Objective 1:
                - Follow-up 1: (if any)
            - Objective 2:
                - Follow-up 1:
                - Follow-up 2:
            - Objective 3:
            ...
    ]

    {few_shot}
    
    Here is a transcript:

    {QA_Sequence}

    Again, be brief, abstract and concise, try to recreate my high-level notes. There are no fixed amount of objectives, 
    but pay attention to which questions are follow-up questions and which are outline-level.
    Write only a few words per outline point.
    '''
    return prompt

def get_outline_only_prompt(outline_text):
    prompt = f'''
    You are an assistant that processes outlines by removing any follow-up sections.

    Please only respond with the provided outline exactly as it is, but exclude any follow-up items.

    Here is the outline:

    ```{outline_text}```

    Here are some examples:
    
    get_Example 1:
    Input:
    [
        Source biography: Jane Doe is a technology expert and author.
        Interview context: The impact of artificial intelligence on modern workplaces.

        - Objective 1: Understanding AI integration in daily operations.
            - Follow-up 1: Challenges faced by employees adapting to AI tools.
        - Objective 2: Ethical considerations of AI deployment.
        - Objective 3: Future trends in AI technology.
            - Follow-up 1: Potential job market shifts due to AI advancements.
    ]

    Output:
    [
        Source biography: Jane Doe is a technology expert and author.
        Interview context: The impact of artificial intelligence on modern workplaces.

        - Objective 1: Understanding AI integration in daily operations.
        - Objective 2: Ethical considerations of AI deployment.
        - Objective 3: Future trends in AI technology.
    ]

    Example 2:
    Input:
    [
        Source biography: John Smith is an environmental scientist.
        Interview context: Climate change effects on coastal regions.

        - Objective 1: Analyzing rising sea levels.
            - Follow-up 1: Impact on local communities.
        - Objective 2: Biodiversity loss in coastal ecosystems.
        - Objective 3: Mitigation strategies for coastal preservation.
            - Follow-up 1: Community-based conservation efforts.
    ]

    Output:
    [
        Source biography: John Smith is an environmental scientist.
        Interview context: Climate change effects on coastal regions.

        - Objective 1: Analyzing rising sea levels.
        - Objective 2: Biodiversity loss in coastal ecosystems.
        - Objective 3: Mitigation strategies for coastal preservation.
    ]
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