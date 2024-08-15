# old taxonomy: coarse-grained and fine-grained question type labels
FINE_GRAINED_TAXONOMY = [
    "kick-off question - initial inquiry",
    "acknowledgement statement - establish empathy",
    "follow-up question - elaboration",
    "follow-up question - probing",
    "follow-up question - re-asking",
    "topic transition question - new topic introduction",
    "topic transition question - segway",
    "opinion and speculation question - opinion seeking",
    "opinion and speculation question - speculative inquiry",
    "verification question - fact-checking",
    "verification question - confirmation",
    "verification question - clarification",
    "challenge question - contradiction",
    "challenge question - critical inquiry",
    "broadening question - scope expansion",
    "other"
]

# new taxonomy: single level of classification while still capturing the essence of each cat/subcat
# include only the topic level labels
TAXONOMY = [
    "initial inquiry question",
    "establishing empathy question",
    "elaboration question",
    "probing question",
    "rephrasing question",
    "topic transition question",
    "opinion seeking question",
    "speculative inquiry question",
    "fact-checking question",
    "confirmation question",
    "clarification question",
    "contradiction challenge question",
    "critical inquiry question",
    "scope expansion question"
]

COURSE_GRAINED_TAXONOMY = [
    "kick-off/ending remark"
    "acknowledgement statement",
    "follow-Up question",
    "topic transition question",
    "opinion/Speculation question",
    "verification question",
    "challenge question", 
    "broadening question"
]

# this prompt instructs LLM to classify the last question in the current interview transcript, given a question type taxonomy
CLASSIFY_USING_TAXONOMY_PROMPT = '''
Here is a comprehensive taxonomy of journalist question type labels and their corresponding motivations:
 - Kick-Off/Ending Remark:
   - Motivation: To initiate or conclude the interview.
 - Acknowledgement Statement:
   - Motivation: To recognize or affirm a previous response or statement made by the interviewee. This can create rapport, demonstrate active listening and empathy.
 - Follow-Up Question:
   - Motivation: To dig deeper into a topic that has already been discussed, seeking further elaboration, or re-phrasing a previous question.
 - Topic Transition Question:
   - Motivation: To smoothly shift the conversation from one subject to another. These questions help maintain the flow of the interview and introduce new topics.
 - Opinion/Speculation Question:
   - Motivation: To solicit the interviewee's personal views or predictions about a subject, often aiming to reveal biases, insights, or potential foresight.
 - Verification Question:
   - Motivation: To confirm the accuracy of a statement, fact, or assumption. This type of question seeks to ensure that information is correct and reliable.
 - Challenge Question:
   - Motivation: To test the interviewee's position, argument, or credibility. These questions are often used to provoke thought, debate, or to highlight inconsistencies.
 - Broadening Question:
   - Motivation: To expand the scope of the discussion, encouraging the interviewee to think about the topic in a broader context or from different perspectives.

Below is the following interview transcript section.

Interview Transcript Section:
{transcript_section}

Here is the last question asked in the transcript section: 
{question}

The format of your response should be in this sequence:
  1. First, repeat the question, then explain your thought process. Pick the single label you think best categorizes the question based on the taxonomy provided above.
  2. Then, return your guess of the question type in brackets.
    Here are some examples of correct label formatting: 
    ex 1. This type of question is: [Initial Inquiry Question]
    ex 2. This type of question is: [Establishing Empathy Question]
    ex 3. This type of question is: [Rephrasing Question]
Don't include the motivation inside the brackets, and don't include multiple labels. Make sure only a single guess for the question type is inside the brackets.
'''

# this prompt instructs LLM to classify the question given a question type taxonomy
CLASSIFY_ALL_QUESTIONS_USING_TAXONOMY_PROMPT = '''
Here is a comprehensive taxonomy of journalist question type labels and their corresponding motivations:
 - Kick-Off/Ending Remark:
   - Motivation: To initiate or conclude the interview.
 - Acknowledgement Statement:
   - Motivation: To recognize or affirm a previous response or statement made by the interviewee. This can create rapport, demonstrate active listening and empathy.
 - Follow-Up Question:
   - Motivation: To dig deeper into a topic that has already been discussed, seeking further elaboration, or re-phrasing a previous question.
 - Topic Transition Question:
   - Motivation: To smoothly shift the conversation from one subject to another. These questions help maintain the flow of the interview and introduce new topics.
 - Opinion/Speculation Question:
   - Motivation: To solicit the interviewee's personal views or predictions about a subject, often aiming to reveal biases, insights, or potential foresight.
 - Verification Question:
   - Motivation: To confirm the accuracy of a statement, fact, or assumption. This type of question seeks to ensure that information is correct and reliable.
 - Challenge Question:
   - Motivation: To test the interviewee's position, argument, or credibility. These questions are often used to provoke thought, debate, or to highlight inconsistencies.
 - Broadening Question:
   - Motivation: To expand the scope of the discussion, encouraging the interviewee to think about the topic in a broader context or from different perspectives.

Below is the following interview transcript:
{transcript}

Here is the question from the transcript I want you to classify using the taxonomy: 
{question}

The format of your response should be in this sequence:
  1. First, repeat the question, then explain your thought process. Pick the single label you think best categorizes the question based on the taxonomy provided above.
  2. Then, return your guess of the question type in brackets.
    Here are some examples of correct label formatting: 
    ex 1. This type of question is: [Initial Inquiry Question]
    ex 2. This type of question is: [Establishing Empathy Question]
    ex 3. This type of question is: [Rephrasing Question]
Don't include the motivation inside the brackets, and don't include multiple labels. Make sure only a single guess for the question type is inside the brackets.
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

    Given these dimensions of similarity as well as the following information below, please evaluate whether the two questions below are overall similar or not. They are either similar or they aren't. 

    Transcript context: {transcript_context}

    Question 1: {LLM_question}
    Question 1 Type Classification: {LLM_question_type}

    Question 2: {human_question}
    Question 2 Type Classification: {Actual_question_type}

    These two questions are two possible continuation questions an interviewer can ask given the current interview so far. In essence, your sole task is to determine whether the intent of these two possible questions are more similar or not different overall.

    Please take things step by step. The format of your response should be in this sequence:
    1. First, repeat the two questions, then explain your thought process comparing these questions across each dimension of similarity. 
    2. Then, answer the following question: In the context of this interview, are the two questions provided more similar or different? 
    Please format your final answer as either "similar" or "different" with brackets. 
    If you think the similarity between the questions are high, please say "similar" instead.
    If you think the similarity between the questions are low, please say "different" instead.
    Your final answer can only be either of the following two: [similar] or [different], not both. 
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
