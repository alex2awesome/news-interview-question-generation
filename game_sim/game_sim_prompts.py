# prompt/instructions for the interviewee
def source_prompt_loader(QA_Sequence, response_type):
    prompt = f'''
    You are a source. You have the following pieces of information but you are not just going to say it all at once, you have to be convinced to give the information. Here is the conversation so far:

    {QA_Sequence}

    Now, respond to the question. Respond with {response_type}:
    '''
    return prompt

# prompt/instructions for interviewer
def interviewer_prompt_loader(QA_Sequence, goals):
    prompt = f'''
    You are an interviewer. Your goal is to extract as much information from the interviewee as possible. 
    
    Here is what you need to know:

    {goals}

    Here is your conversation so far:

    {QA_Sequence}

    Now, ask the next question.
    '''

# prompt that summarizes the interview transcript into key information items that the source has
def extraction_prompt_loader(QA_Sequence):
    prompt = f"""
    You are an AI tasked with extracting key pieces of information from an interview transcript. Below is the transcript:

    {QA_Sequence}

    Please extract the key pieces of information provided by the interviewee, formatted as follows:
    - Information item #1
    - Information item #2
    - Information item #3
    …
    """
    return prompt