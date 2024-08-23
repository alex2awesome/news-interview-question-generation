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
    "follow-up question",
    "topic-transition question",
    "opinion/speculation question",
    "verification question",
    "challenge question", 
    "broadening question"
]

OLD_DEFINITIONS = '''
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
   - Definition: Solicits the interviewee's personal views or predictions about a subject. Can reveal biases and insights.
 - Challenge Question:
   - Definition: Tests the interviewee's position, argument, or credibility. These questions are often used to provoke thought, debate, or to highlight inconsistencies.
 - Broadening Question:
   - Definition: To expand the scope of the discussion, encouraging the interviewee to think about the topic in a broader context or from different perspectives.
'''

DEFINITIONS = '''
 - Starting/Ending Remarks:
   - Definition: Initiates or concludes the interview. Often not be in the form of a question.
 - Outline-Level Question:
   - Definition: Introduces a top-level topic into the conversation. Shifts the conversation from one subject to another. These questions are evidence of outline-level goals in the interview that the journalist wishes to ask, not simply responding to previous questions.
 - Acknowledgement Statement:
   - Definition: Affirms the interviewee, often by explicitly affirming the interviewee's previous response. This can create rapport, demonstrate active listening and empathy.
 - Follow-Up Question:
   - Definition: Digs deeper into a topic being discussed, seeks further elaboration, or re-phrases a previous question in a way that keeps the interview on the same topic.
 - Verification Question:
   - Definition: Confirms the accuracy of a statement, fact, or assumption. This type of question seeks to ensure that information is correct and reliable.
 - Opinion/Speculation Question:
   - Definition: Solicits the interviewee's personal views or predictions about a subject. Can reveal biases and insights.
 - Challenge Question:
   - Definition: Tests the interviewee's position, argument, or credibility. These questions are often used to provoke thought, debate, or to highlight inconsistencies.
 - Broadening Question:
   - Definition: To expand the scope of the discussion, encouraging the interviewee to think about the topic in a broader context or from different perspectives.
'''

OLD_FEW_SHOT_EXAMPLES = '''
  Previous Question Context: The economic impact of a newly implemented policy or mandate.
  Question: Can you explain more about how the mandate is hurting the economy?
  Response:
  The question seeks to dive deeper into a topic and get more information.
  [Follow-Up Question]
  
  Previous Question Context: A starting remark introducing the interviewee and background.
  Question: Now I want to talk about Syria. Can you explain how your work in Aleppo changed your career?
  Response:
  The previous question (starting remark) was the first question in the interview. Typically, the question after the starting remark is a topic-transition question. We verify this is indeed the case, as the topic shifted from (topic A) an introduction of the interviewee to (topic B) Syria and how his work there impacted his life.
  [Topic-Transition Question]

  Previous Question Context: Presidential debate between Donald Trump and Hillary Clinton.
  Question: Let's look forward to the vice presidential debate. This is happening Tuesday. Mike Pence, Tim Kaine will go head to head. We haven't heard a whole lot from either of them so far. Do you think they're just going to echo what their running mates have been saying?
  Response: 
  The topic has shifted from (topic A) the presidential debate to (topic B) the vice presidential debate. After some context, the interviewer then asks for the interviewee's opinion.
  [Topic Transition Question, Opinion/Speculation Question]

  Previous Question Context: Discussion on the ongoing handling of the COVID-19 pandemic by various government administrations.
  Question: Do you believe the current administration is handling the pandemic well?
  Response:
  The question appears to be asking for an opinion rather than a set of facts.
  [Opinion/Speculation Question]

  Previous Question Context: The interviewee has just claimed that there will be a rise in unemployment in the next decade years.
  Can you provide evidence to support that claim?
  Response:
  The journalist is asking for further details specifically to back up a previous remark.
  [Verification Question]
'''

FEW_SHOT_EXAMPLES = '''
  Previous Question Context: The economic impact of a newly implemented policy or mandate.
  Question: Can you explain more about how the mandate is hurting the economy?
  Response:
  The question seeks to dive deeper into a topic and get more information.
  [Follow-Up Question]
  
  Previous Question Context: A starting remark introducing the interviewee and background.
  Question: Now I want to talk about Syria. Can you explain how your work in Aleppo changed your career?
  Response:
  The previous question (starting remark) was the first question in the interview. Typically, the question after the starting remark is a topic-transition question. We verify this is indeed the case, as the topic shifted from (topic A) an introduction of the interviewee to (topic B) Syria and how his work there impacted his life.
  [Outline-Level Question]

  Previous Question Context: Presidential debate between Donald Trump and Hillary Clinton.
  Question: Let's look forward to the vice presidential debate. This is happening Tuesday. Mike Pence, Tim Kaine will go head to head. We haven't heard a whole lot from either of them so far. Do you think they're just going to echo what their running mates have been saying?
  Response: 
  The topic has shifted from (topic A) the presidential debate to (topic B) the vice presidential debate. After some context, the interviewer then asks for the interviewee's opinion.
  [Outline-Level Question, Opinion/Speculation Question]

  Previous Question Context: Discussion on the ongoing handling of the COVID-19 pandemic by various government administrations.
  Question: Do you believe the current administration is handling the pandemic well?
  Response:
  The question appears to be asking for an opinion rather than a set of facts.
  [Opinion/Speculation Question]

  Previous Question Context: The interviewee has just claimed that there will be a rise in unemployment in the next decade years.
  Question: Can you provide evidence to support that claim?
  Response:
  The journalist is asking for further details specifically to back up a previous remark.
  [Verification Question]

  Previous Question Context: The interviewee has just talked about brain size and intelligence.
  Question: Overall, what's the importance of this right now? Why - this debate has been framed by some as beyond the pale, we shouldn't even discuss it. Do you think, first of all, that it is a distraction to discuss it, something that's really important for us to revisit?
  Response: 
  The journalist is encouraging the source to place these remarks in the context of a broader conversation of how it affects society.
  [Broadening Question]

  Previous Question Context: The interviewer has just started the interview, introduced the interviewee as a tour guide operator in Afghanistan, and discussed a recent attack.
  Question: We should first say the minibus that was recently attacked was not one of your tours. But you have led groups in Afghanistan, and I just have to ask why.
  Response: 
  The journalist is following up on the introductory remarks, and introducing a new focus on the interviewee.
  [Outline-Level Question, Follow-Up Question]
'''

OLD_FORMAT = '''
The format of your response should be in this sequence:
  1. First, explain your thought process step by step. How does the given question relate to the previous question? Does this question follow in the same overall topic as the previous question/remark or does it start a new topic? 
  2. Then pick the single label, or labels, you think best categorize the question, based on the schema above.
  3. Finally, return your guess of the question type, in brackets.
Don't include the definition inside the brackets.
'''

FORMAT = '''
Respond in this way:
  1. First, break this down, step-by-step:
    * What are the primary discussion points the journalist wishes to have during the conversation?
    * How does the current question relate to the previous question? 
    * Does the current question shift the focus in some way from the previous question/remark and shift to a different primary discussion point? (If yes, then this is may be an Outline-Level Question or Broadening Question)
    * Or does continue in the same line of inquiry? (If yes, then this is probably any of the others)
  2. Based on this reasoning, pick the single label, or labels, you think best categorize the question, based on the schema above. 
  3. Return the labels you select as a comma-separated list INSIDE brackets. Return the reasoning in part 1 BEFORE the brackets.
'''

# this prompt instructs LLM to classify the last question in the current interview transcript, given a question type taxonomy
def get_classify_taxonomy_prompt(transcript_section, question):
      prompt = f'''
      I am trying to understand the kinds of questions asked by journalists. 
      I will show you the question the journalist asks. I will also show you the conversational history between the journalist (interviewer) and source (interviewee) for context.
      Please label the question according to the following 8 categories of questions we've identified.

      Here are the schema categories:

      {DEFINITIONS}
      
      {FORMAT}

      Here are some examples (here, I show just the previous question context and the given question to save space):

      {FEW_SHOT_EXAMPLES}

      Now it's your turn.

      Below is the interview transcript:
      {transcript_section}

      Here is the next question asked, which you will classify: 
      Question: {question}
      Response:
      '''
      return prompt

# this prompt instructs LLM to classify the question given a question type taxonomy
def get_classify_all_questions_taxonomy_prompt(transcript, question):
      prompt = f'''
      I am trying to understand the kinds of questions asked by journalists. 
      I will show you the transcript between the journalist (interviewer) and source (interviewee). I will then ask about a specific question in that transcript.
      Please label the question according to the following 8 categories of questions we've identified.

      Here are the schema categories:

      {DEFINITIONS}

      Here are some examples of questions, a summary of prior context, and the schema categories they belong to:

      ```{FEW_SHOT_EXAMPLES}```

      Ok, now it's your turn. Here is the interview transcript:
      ```{transcript}```

      And here is the question from the transcript I want you to classify using the taxonomy: 
      Question: ```{question}```
      Now it's your turn.

      {FORMAT}
      
      Please respond now:
      '''
      return prompt

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


if __name__ == "__main__": 
      transcript = "<transcript>"
      question = "<question>"
      print(get_classify_all_questions_taxonomy_prompt(transcript, question))