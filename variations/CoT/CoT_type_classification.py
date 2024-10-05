import os
import sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import pandas as pd
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluators.vllm_type_classification import classify_LLM_question_process_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


if __name__ == "__main__":
    dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/CoT/LLM_human_question_labels_classified.csv"
    other_dataset_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/CoT/QA_Seq_LLM_generated.csv"   
    if os.path.exists(dataset_path):
      print("LLM_human_question_labels_classified found!")
      LLM_questions_df = pd.read_csv(dataset_path)
      print(LLM_questions_df)

      type_classified_df = classify_LLM_question_process_dataset(LLM_questions_df, output_dir="output_results/CoT", batch_size=100, model_name="meta-llama/Meta-Llama-3-70B-Instruct") # saves type_classification labels in LLM_classified_results.csv
      print(type_classified_df)
    else:
      print("LLM_human_question_labels_classified NOT found, using QA_Seq_LLM_generated!")
      LLM_questions_df = pd.read_csv(other_dataset_path)
      print(LLM_questions_df)

      type_classified_df = classify_LLM_question_process_dataset(LLM_questions_df, output_dir="output_results/CoT", batch_size=100, model_name="meta-llama/Meta-Llama-3-70B-Instruct") # saves type_classification labels in LLM_classified_results.csv
      print(type_classified_df)

      saved_to_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/CoT/LLM_classified_results.csv"
      print(f"os.path.exists: {os.path.exists(saved_to_path)}")

      temp_df = pd.read_csv(dataset_path)
      human_question_labels = temp_df['Actual_Question_Type']
      type_classified_df['Actual_Question_Type'] = human_question_labels
      type_classified_df.to_csv(saved_to_path, index=False)
      print(f"csv file saved to {saved_to_path}")

    # checked that 8B model works? y/n: y (validated by michael)
