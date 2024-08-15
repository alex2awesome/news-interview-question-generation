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

# course grained taxonomy
TAXONOMY = [
    "starting/ending remarks"
    "acknowledgement statement",
    "follow-Up question",
    "topic transition question",
    "opinion/speculation question",
    "verification question",
    "challenge question", 
    "broadening question"
]

DEFINITIONS = '''
 - Starting/Ending Remarks:
   - Definition: Initiates or concludes the interview. Often not be in the form of a question.
 - Acknowledgement Statement:
   - Definition: Affirms the interviewee, often by explicitly affirming the interviewee's previous response. This can create rapport, demonstrate active listening and empathy.
 - Follow-Up Question:
   - Definition: Digs deeper into a topic being discussed, seeks further elaboration, or re-phrases a previous question in a way that keeps the interview on the same topic.
 - Verification Question:
   - Definition: Confirms the accuracy of a statement, fact, or assumption. This type of question seeks to ensure that information is correct and reliable.
 - Topic-Transition Question:
   - Definition: Shifts the conversation from one subject to another. These questions introduce new topics into the interview, and are evidence of outline-level goals in the interview.
 - Opinion/Speculation Question:
   - Definition: Solicits the interviewee's personal views or predictions about a subject. Can revealing biases and insights.
 - Challenge Question:
   - Definition: Tests the interviewee's position, argument, or credibility. These questions are often used to provoke thought, debate, or to highlight inconsistencies.
 - Broadening Question:
   - Definition: To expand the scope of the discussion, encouraging the interviewee to think about the topic in a broader context or from different perspectives.
'''

FEW_SHOT_EXAMPLES = '''
  Question: Can you explain more about how the mandate is hurting the economy?
  Response:
  The question seeks to dive deeper into a topic and get more information.
  [Follow-Up Question]

  Question: Now I want to talk about Syria. Can you explain how your work in Aleppo changed your career?
  Response:
  It appears that Syria is a new topic, since the interview is shifting.
  [Topic-Transition Question]


  Question: Do you believe the current administration is handling the pandemic well?
  Response:
  The question appears to be asking for an opinion rather than a set of facts.
  [Opinion/Speculation Question]


  Can you provide evidence to support that claim?
  Response:
  The journalist is asking for further details specifically to back up a previous remark.
  [Verification Question]
'''

FORMAT = '''
The format of your response should be in this sequence:
  1. First, explain your thought process given the question. Pick the single label, or labels, you think best categorize the question, based on the taxonomy above.
  2. Then, return your guess of the question type, in brackets.
Don't include the motivation inside the brackets.
'''

# this prompt instructs LLM to classify the last question in the current interview transcript, given a question type taxonomy
CLASSIFY_USING_TAXONOMY_PROMPT = f'''
I am trying to understand the kinds of questions asked by journalists. 
I will show you the question the journalist asks. I will also show you the conversational history between the journalist and source for context.
Please label the question according to the following 8 categories of questions we've identified.

Here are the schema categories:

{DEFINITIONS}

{FORMAT}

Here are some examples (here, I show just the questions to save space):

{FEW_SHOT_EXAMPLES}

Now it's your turn.

Below is the interview transcript:
{{transcript_section}}

Here is the next question asked, which you will classify: 
Question: {{question}}
Response:
'''

# this prompt instructs LLM to classify the question given a question type taxonomy
CLASSIFY_ALL_QUESTIONS_USING_TAXONOMY_PROMPT = f'''
I am trying to understand the kinds of questions asked by journalists. 
I will show you the conversational history between the journalist and source. I will then ask about a specific question in that history.
Please label the question according to the following 8 categories of questions we've identified.

Here are the schema categories:

{DEFINITIONS}

{FORMAT}

Here are some examples (here, I show just the questions to save space):

{FEW_SHOT_EXAMPLES}

Now it's your turn.

Below is the following interview transcript:
{{transcript}}

Here is the question from the transcript I want you to classify using the taxonomy: 
Question: {{question}}
Response:
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
