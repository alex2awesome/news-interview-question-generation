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
      Example: 'What advice would you give to Americans who haven’t chosen to get vaccinated?'
Acknowledgement Statement:
  - Establish Empathy.
      Example: 'Well, this must have been pretty astonishing?'
Follow-Up Questions:
  - Elaboration: Asking the interviewee to provide more details on a specific point.
      Example: 'Can you explain more about how the mandate is hurting the economy?'
  - Probing: Digging deeper into a topic to uncover more information or hidden insights.
      Example: 'Why do you think the administration is not promoting therapeutics as much as vaccines?'
  - Re-asking: Rephrasing a question to get a direct answer if the initial response was unsatisfactory.
      Example: 'But what specific advice do you have about the vaccines themselves?'
Topic Transition Questions:
  - New Topic Introduction: Moving on to a completely new topic.
      Example: 'Now, let’s talk about the upcoming election…'
  - Segway: Smoothly transitioning from one related topic to another.
      Example: 'Speaking of the economy, how do you see the job market evolving next year?'
Opinion and Speculation Questions:
  - Opinion Seeking: Asking for the interviewee’s personal views or opinions.
      Example: 'Do you believe the current administration is handling the pandemic well?'
  - Speculative Inquiry: Asking the interviewee to speculate or predict future events.
      Example: 'What do you think will happen if the mandates continue?'
Verification Questions:
  - Fact-Checking: Verifying the accuracy of a statement or claim made by the interviewee.
      Example: 'Can you provide evidence to support that claim?'
  - Confirmation: Confirming an understanding or interpretation of the interviewee’s previous statements.
      Example: 'So, you’re saying the therapeutics should be prioritized over the vaccines?'
  - Clarification: Seeking to clarify a vague or incomplete answer.
      Example: 'In other words, are you saying X?'
Challenge Questions:
  - Contradiction: Pointing out inconsistencies or contradictions in the interviewee’s statements.
      Example: 'You mentioned the mandate is harmful, but earlier you supported vaccination efforts. Can you explain this?'
  - Critical Inquiry: Critically questioning the interviewee’s stance or actions.
      Example: 'Given the economic impact, do you think your administration’s policies were effective?'
Broadening Questions:
  - Scope Expansion: Expanding the scope of the interview to include more general or additional topics.
      Example: 'How do you see these policies affecting global markets?'

Below is the following interview transcript section.

Interview Transcript Section:
{transcript_section}

Here is the last question asked in the transcript section: {question}
Please classify the type of this question, based on the taxonomy provided.
First explain your thought process, then return your guess of the question type in brackets (both the category and the subcategory). For example: [Acknowledgement Statement - Establish empathy]
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

    Given the following information as well as the dimensions of similarity provided above, please evaluate whether the two questions below are similar or not. They are either similar or they aren't. The two questions are two possible continuation questions an interviewer can ask given the current interview transcript (conversation so far):

    Transcript context: {transcript_context}

    Question 1: {LLM_question}
    Question 1 Type Classification: {LLM_question_type}

    Question 2: {human_question}
    Question 2 Type Classification: {Actual_question_type}

    Please take things step by step by first explaining your thought process for each dimension. 
    Finally, answer the following question: In the context of this interview, are the two questions provided more similar or different? 
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

# this prompt instructs LLM to predict/guess the next question asked by the interviewer given a QA_sequence
LLM_QUESTION_GENERATOR_PROMPT = '''
Here's the dialogue so far between me, the interviewer, and the guest:

{QA_Sequence}

I would like you to guess the next question I will ask, however before that, let's first take things step by step. 
Analyze the last piece of dialogue the guest has said. 
Now think about what I as an interviewer could be thinking about in response to that. 
What are the possible motivations for the kinds of questions I could be asking?

Now, please guess the next question I will ask. Format your final guess for the question in brackets 
like this: [Guessed Question]. 
Next, please explain the motivation behind the question you provided in paragraph form, then format it with parentheses like this: (motivation explanation)
'''