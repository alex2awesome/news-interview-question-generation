TAXONOMY = [
    "Kick-Off Questions - Initial Inquiry",
    "Kick-Off Questions - Acknowledgement Statement",
    "Follow-Up Questions - Elaboration",
    "Follow-Up Questions - Probing",
    "Follow-Up Questions - Re-asking",
    "Topic Transition Questions - New Topic Introduction",
    "Topic Transition Questions - Segway",
    "Opinion and Speculation Questions - Opinion Seeking",
    "Opinion and Speculation Questions - Speculative Inquiry",
    "Verification Questions - Fact-Checking",
    "Verification Questions - Confirmation",
    "Verification Questions - Clarification",
    "Challenge Questions - Contradiction",
    "Challenge Questions - Critical Inquiry",
    "Broadening Questions - Scope Expansion"
]

CLASSIFY_USING_TAXONOMY_PROMPT = '''
Here is a comprehensive taxonomy of journalist question types/motivations:
Kick-Off Questions:
  - Initial Inquiry: Asking for basic information on a topic.
      Example: 'What advice would you give to Americans who haven’t chosen to get vaccinated?'
Acknowledgement Statement:
  - Establish empathy.
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

Given the following interview transcript section, please look and classify the type of the last question asked.

Interview Transcript Section:
{transcript_section}

Please first explain your thought process, then return your guess of the question type in brackets (both the category and the subcategory). For example: [Acknowledgement Statement - Establish empathy]
'''

DIMENSION_OF_SIMILARITY_PROMPT = '''
Evaluate the consistency between the following two questions based on four dimensions:
    1. Informational: Do the questions target the same specific information or facts?
    2. Motivational: Do the questions have the same motivation or underlying purpose?
    3. Contextual: Are both questions equally appropriate for the specific context provided?
    4. Stylistic: Do the questions have similar styles in terms of tone, complexity, and structure?

    Please label each dimension with either "yes" or "no".

    Transcript Context: {transcript_context}

    LLM Question: {llm_question}

    Human Question: {human_question}

    Please take things step by step by first explaining your thought process for each dimension. 
    Then, return your label for each dimension in the following format: [Informational label, Motivational label, Contextual label, Stylistic label]
    
    Example 1: [Yes, Yes, No, Yes]
    Example 2: [No, No, No, No]
    Example 3: [Yes, No, Yes, No]
'''
