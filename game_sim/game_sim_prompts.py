# prompt/instructions for the interviewee
def source_prompt_loader(QA_Sequence, response_type):
    prompt = f'''
    You are a source. You have the following pieces of information but you are not just going to say it all at once, you have to be convinced to give the information. 
    
    Here is the conversation so far:

    {QA_Sequence}

    You're generally {response_type}. Choose whether you want to respond to the question, and then give your repsonse. Respond with {response_type}:
    '''
    return prompt
    # plug in the info_items
    # for response type you might really need to define what each means
    # make sure the source profile isn't just 100% being evasive, or confused, that's not useful. We want to make sure it's realistic, even if that means we do some rule-based behaviors. maybe we want some set of frequency of specified behavior - eg. response_type = evasive --> 40% evasive, 60% straightforward.
    # i.e youre super anxious, you will only respond honestly after a lot of acknowledgment, etc.

# prompt/instructions for interviewer
def interviewer_prompt_loader(QA_Sequence, objectives):
    prompt = f'''
    You are an interviewer. Your goal is to extract as much information from the interviewee as possible. 
    
    Here is what you need to know:

    {objectives}

    Here is your conversation so far:

    {QA_Sequence}

    Now, ask the next question.
    '''

# prompt that summarizes the interview transcript into key information items that the source has
def extraction_prompt_loader(QA_Sequence):
    prompt = f"""
    You are an AI tasked with extracting key pieces of information from an interview transcript. 
    
    Below is the transcript:

    {QA_Sequence}

    Please extract the key pieces of information provided by the interviewee, formatted as follows:
    - Information item #1
    - Information item #2
    - Information item #3
    …
    """
    return prompt