# baseline_type_classification.py

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
Here is a comprehensive taxonomy of journalist question type labels and their corresponding motivations:
  -	Initial Inquiry Question: 
      - motivation: Asking for basic information on a topic.
	-	Establishing Empathy Question: 
      - motivation: Making an acknowledgment statement to build rapport.
	-	Elaboration Question: 
      - motivation: Asking for more details on a specific point.
	-	Probing Question: 
      - motivation: Digging deeper to uncover more information or hidden insights.
	-	Rephrasing Question: 
      - motivation: Re-asking or rephrasing a question to obtain a clearer or more direct answer
	-	Topic Transition Question: 
      - motivation: Moving on to a new topic or smoothly transitioning between related topics.
	-	Opinion Seeking Question: 
      - motivation: Asking for personal views or opinions.
	-	Speculative Inquiry Question: 
      - motivation: Requesting predictions or speculation about future events.
	-	Fact-Checking Question: 
      - motivation: Verifying the accuracy of a statement or claim.
	-	Confirmation Question: 
      - motivation: Confirming an understanding or interpretation of previous statements.
	-	Clarification Question: 
      - motivation: Seeking to clarify a vague or incomplete answer.
	-	Contradiction Challenge Question: 
      - motivation: Pointing out inconsistencies or contradictions.
	-	Critical Inquiry Question: 
      - motivation: Critically questioning the interviewee’s stance or actions.
	-	Scope Expansion Question: 
      - motivation: Broadening the discussion to include additional or more general topics.

Below is the following interview transcript section.

Interview Transcript Section:
{transcript_section}

Here is the last question asked in the transcript section: 
{question}

The format of your response should be in this sequence:
  1. First, repeat the question, then explain your thought process. Pick the single label you think best categorizes the question based on the taxonomy provided above.
  2. Then, return your guess of the question type in brackets (both the category and the subcategory).
    Here are some examples of correct label formatting: 
    ex 1. This type of question is: [Initial Inquiry Question]
    ex 2. This type of question is: [Establishing Empathy Question]
    ex 3. This type of question is: [Rephrasing Question]
Don't include the motivation inside the brackets, and don't include multiple labels. Make sure only a single guess for the question type is inside the brackets.
'''

if __name__ == "__main__":
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/baseline/QA_Seq_LLM_generated.csv"
    LLM_questions_df = pd.read_csv(dataset_path)
    print(LLM_questions_df)

    type_classified_df = classify_question_process_dataset(LLM_questions_df, output_dir="output_results/baseline", batch_size=40, model_name="meta-llama/Meta-Llama-3-70B-Instruct") # saves type_classification labels in LLM_classified_results.csv
    print(type_classified_df)

    # checked that 8B model works? y/n: y (validated by michael)
