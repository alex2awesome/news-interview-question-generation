# question type taxonomy
TAXONOMY = [
    "kick-off questions - initial inquiry",
    "acknowledgement statement - establish empathy",
    "follow-up questions - elaboration",
    "follow-up questions - probing",
    "follow-up questions - re-asking",
    "topic transition questions - new topic introduction",
    "topic transition questions - segway",
    "opinion and speculation questions - opinion seeking",
    "opinion and speculation questions - speculative inquiry",
    "verification questions - fact-checking",
    "verification questions - confirmation",
    "verification questions - clarification",
    "challenge questions - contradiction",
    "challenge questions - critical inquiry",
    "broadening questions - scope expansion",
    "other"
]

# this prompt instructs LLM to classify the last question in the current interview transcript, given a question type taxonomy
CLASSIFY_USING_TAXONOMY_PROMPT = '''
Here is a comprehensive taxonomy of journalist question types/motivations:

Kick-Off Questions:
  - Initial Inquiry: Asking for basic information on a topic.
Acknowledgement Statement:
  - Establish Empathy.
Follow-Up Questions:
  - Elaboration: Asking the interviewee to provide more details on a specific point.
  - Probing: Digging deeper into a topic to uncover more information or hidden insights.
  - Re-asking: Rephrasing a question to get a direct answer if the initial response was unsatisfactory.
Topic Transition Questions:
  - New Topic Introduction: Moving on to a completely new topic.
  - Segway: Smoothly transitioning from one related topic to another.
Opinion and Speculation Questions:
  - Opinion Seeking: Asking for the interviewee’s personal views or opinions.
  - Speculative Inquiry: Asking the interviewee to speculate or predict future events.
Verification Questions:
  - Fact-Checking: Verifying the accuracy of a statement or claim made by the interviewee.
  - Confirmation: Confirming an understanding or interpretation of the interviewee’s previous statements.
  - Clarification: Seeking to clarify a vague or incomplete answer.
Challenge Questions:
  - Contradiction: Pointing out inconsistencies or contradictions in the interviewee’s statements.
  - Critical Inquiry: Critically questioning the interviewee’s stance or actions.
Broadening Questions:
  - Scope Expansion: Expanding the scope of the interview to include more general or additional topics.

Below is the following interview transcript section.

Interview Transcript Section:
{transcript_section}

Here is the last question asked in the transcript section: 
{question}

The format of your response should be in this sequence:
1. First, explain your thought process. Consider all possible answers, then pick the one you think best categorizes the question based on the taxonomy provided.
2. Then, return your guess of the question type in brackets (both the category and the subcategory). For example: [Acknowledgement Statement - Establish empathy]
'''

# this prompt instructs LLM to classify the question given a question type taxonomy
CLASSIFY_ALL_QUESTIONS_USING_TAXONOMY_PROMPT = '''
Here is a comprehensive taxonomy of journalist question types/motivations:
Kick-Off Questions:
  - Initial Inquiry: Asking for basic information on a topic.
Acknowledgement Statement:
  - Establish Empathy: Acknowledges what the guest is saying, doesn't need to be a question.
Follow-Up Questions:
  - Elaboration: Asking the interviewee to provide more details on a specific point.
  - Probing: Digging deeper into a topic to uncover more information or hidden insights.
  - Re-asking: Rephrasing a question to get a direct answer if the initial response was unsatisfactory.
Topic Transition Questions:
  - New Topic Introduction: Moving on to a completely new topic.
  - Segway: Smoothly transitioning from one related topic to another.
Opinion and Speculation Questions:
  - Opinion Seeking: Asking for the interviewee’s personal views or opinions.
  - Speculative Inquiry: Asking the interviewee to speculate or predict future events.
Verification Questions:
  - Fact-Checking: Verifying the accuracy of a statement or claim made by the interviewee.
  - Confirmation: Confirming an understanding or interpretation of the interviewee’s previous statements.
  - Clarification: Seeking to clarify a vague or incomplete answer.
Challenge Questions:
  - Contradiction: Pointing out inconsistencies or contradictions in the interviewee’s statements.
  - Critical Inquiry: Critically questioning the interviewee’s stance or actions.
Broadening Questions:
  - Scope Expansion: Expanding the scope of the interview to include more general or additional topics.

Here is the following interview transcript:
{transcript}

Here is a question I want you to classify from the transcript: {question}
Please classify the type of this question, based on the taxonomy provided above. 
First explain your thought process, then return your guess of the question type in brackets (both the category and the subcategory from the taxonomy). Make sure your category and subcategory follows from the taxonomy exactly.
    For example: [Acknowledgement Statement - Establish empathy]
If you believe that the type of the question is not in the current taxonomy, please format your answer as: [Other]
'''

# this prompt instructs LLM to evaluate two different questions based on dimensions of similarity
WIP_DIMENSION_OF_SIMILARITY_PROMPT = '''
Evaluate the consistency between the following two questions based on four dimensions:
    1. Informational: Do the questions target the same specific information or facts?
    2. Motivational: Do the questions have the same motivation or underlying purpose?
    3. Contextual: Are both questions equally appropriate for the specific context provided?
    4. Stylistic: Do the questions have similar styles in terms of tone, complexity, and structure?

    Please label each dimension with either "yes" or "no".

    Transcript Context: {transcript_context}

    LLM Question: {LLM_question}

    Human Question: {human_question}

    Please take things step by step by first explaining your thought process for each dimension. 
    Then, return your label for each dimension in the following format as your final answer: [Informational label, Motivational label, Contextual label, Stylistic label]

    Example 1: [Yes, Yes, No, Yes]
    Example 2: [No, No, No, No]
    Example 3: [Yes, No, Yes, No]

    Make sure that only your final answer has brackets.
'''

# this prompt instructs LLM to evaluate two different questions based on dimensions of similarity
DIMENSION_OF_SIMILARITY_PROMPT = '''
Dimensions of Similarity:
    1. Informational: Do the questions target the same specific information or facts?
    2. Motivational: Do the questions have the same motivation or underlying purpose?
    3. Contextual: Are both questions equally appropriate for the specific context provided?
    4. Stylistic: Do the questions have similar styles in terms of tone, complexity, and structure?

    Given these dimensions of similarity as well as the following information below, please evaluate whether the two questions below are similar or not. They are either similar or they aren't. The two questions are two possible continuation questions an interviewer can ask given the current interview so far.

    Transcript context: {transcript_context}

    Question 1: {LLM_question}
    Question 1 Type Classification: {LLM_question_type}

    Question 2: {human_question}
    Question 2 Type Classification: {Actual_question_type}

    Please take things step by step. The format of your response should be in this sequence:
    1. First, explain your thought process for each dimension. 
    2. Then, answer the following question: In the context of this interview, are the two questions provided more similar or different? 
    
    Please format your final answer as either similar or different with brackets: [similar] or [different]
    Make sure that only your final answer has brackets.
'''

# this prompt is for generating additional context given the entire transcript
CONTEXT_GENERATOR_PROMPT = '''
Please read over this transcript. Write the following information in a brief paragraph:
    Introduce the guest you are interviewing
    Identify the purpose of the interview
    Identify the guest’s involvement in the interview topic and his/her goals.

Transcript: {transcript}   
'''

# (QASeq only) baseline variation: motivation is asked afterwards so that it doesn't affect the question generated (!= CoT)
BASELINE_LLM_QUESTION_PROMPT = '''
Your task is to predict the next question that will follow in an interview.
Make sure that you are recognizing the interviewee's last comment and acknowledging it when appropriate, rather than immediately moving on and asking a question. When you do decide acknowledgment is necessary, make sure your response is personal and empathetic (sound like you care about what the interviewee has to say). This can simply be acknowledging what they said.

The format of your response should be in this sequence:
1. First, guess the next question asked by the interviewer. Format your final guess for the question in brackets like this: [Guessed Question]. 
2. Then, explain the main motivation/intent behind the question that should be asked, then format your explanation with parentheses like this: (motivation explanation)

Here is the interview so far:
{QA_Sequence}

Remember to format your guess for the next question the interviewer asks in brackets [], then your motivation explanation in parentheses ().
'''

# (QASeq + CoT) Chain of Thought variation: motivation is asked before the question to influence the question generated
CoT_LLM_QUESTION_PROMPT = '''
Your task is to predict the next question that will follow in an interview.
Make sure that you are recognizing the interviewee's last comment and acknowledging it when appropriate, rather than immediately moving on and asking a question. When you do decide acknowledgment is necessary, make sure your response is personal and empathetic (sound like you care about what the interviewee has to say). This can simply be acknowledging what they said.

Think about this step by step. For the following questions, write out your thoughts:
  - How did the previous response of the interview address the question?
  - Did they answer the question or do we need to ask a clarifying question?
  - What other components does this story need?/What more information does this source have?
  - Do we need ask a follow up?

The format of your response should be in this sequence:
1. First, write out your thinking (in whatever format you want)
2. Next, explain the main motivation/intent behind the question that should be asked, then format your explanation with parentheses like this: (motivation explanation)
3. Lastly, guess the next question asked by the interviewer. Format your final guess for the question in brackets like this: [Guessed Question]. 

Here is the interview so far:
{QA_Sequence}

Remember to format your motivation in parentheses (), then your guess for the next question asked in brackets [].
'''

# (QASeq + Outline) variation: additional context is provided along with the QA Sequence
OUTLINE_LLM_QUESTION_PROMPT = '''
Your task is to predict the next question that will follow in an interview. I will give you the current interview dialogue as well as the motivation behind the interview.
Make sure that you are recognizing the interviewee's last comment and acknowledging it when appropriate, rather than immediately moving on and asking a question. When you do decide acknowledgment is necessary, make sure your response is personal and empathetic (sound like you care about what the interviewee has to say). This can simply be acknowledging what they said.

The format of your response should be in this sequence:
1. First, guess the next question asked by the interviewer. Format your final guess for the question in brackets like this: [Guessed Question]. 
2. Then, explain the main motivation/intent behind the question that should be asked, then format your explanation with parentheses like this: (motivation explanation)

Here is the relevant information:
{outline_statement}

Here is an outline of your goals and top questions you want to ask for the interview:
{interview_goals}

{general_questions}

Here is the interview so far:
{QA_Sequence}

Remember to format your guess for the next question asked in brackets [], then your motivation in parentheses ().
'''

# (QASeq + CoT + Outline) variation: additional context is provided along with the QA Sequence and chain-of-thought technique
CoT_OUTLINE_LLM_QUESTION_PROMPT = '''
Your task is to predict the next question that will follow in an interview. I will give you the current interview dialogue as well as the motivation behind the interview.
Make sure that you are recognizing the interviewee's last comment and acknowledging it when appropriate, rather than immediately moving on and asking a question. When you do decide acknowledgment is necessary, make sure your response is personal and empathetic (sound like you care about what the interviewee has to say). This can simply be acknowledging what they said.

Think about this step by step. For the following questions, write out your thoughts:
  - How did the previous response of the interview address the question?
  - Did they answer the question or do we need to ask a clarifying question?
  - What other components does this story need?/What more information does this source have?
  - Do we need ask a follow up?

The format of your response should be in this sequence:
1. First, write out your thinking (in whatever format you want)
2. Next, explain the main motivation/intent behind the question that should be asked, then format your explanation with parentheses like this: (motivation explanation)
3. Lastly, guess the next question asked by the interviewer. Format your final guess for the question in brackets like this: [Guessed Question]. 

Here is the relevant information:
{outline_statement}

Here is an outline of your goals and top questions you want to ask for the interview:
{interview_goals}

{general_questions}

Here is the interview so far:
{QA_Sequence}

Remember to format your motivation in parentheses (), and your guess for the next question asked in brackets [].
'''
