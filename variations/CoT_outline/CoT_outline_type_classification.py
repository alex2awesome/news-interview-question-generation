import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluators.vllm_type_classification import classify_question_process_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# this prompt instructs LLM to classify the last question in the current interview transcript, given a question type taxonomy
'''
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

if __name__ == "__main__":
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/CoT_outline/QA_Seq_LLM_generated.csv"
    LLM_questions_df = pd.read_csv(dataset_path)

    type_classified_df = classify_question_process_dataset(LLM_questions_df, output_dir="output_results/CoT_outline", batch_size=50, model_name="meta-llama/Meta-Llama-3-70B-Instruct") # saves type_classification labels in LLM_classified_results.csv
    print(type_classified_df)

    # checked that 8B model works? y/n: y (validated by michael)
    