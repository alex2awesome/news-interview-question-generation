import pandas as pd
import os
import sys
import random
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import combine_N_qa_pairs_and_next_question

def count_qa_pairs(speaker):
    # remove trailing 'host' entries
    while speaker and 'host' in speaker[-1].lower():
        speaker.pop()
    
    qa_pair_count = 0
    current_speaker = None

    for spk in speaker:
        if 'host' in spk.lower() and current_speaker != 'host':
            qa_pair_count += 1
        current_speaker = 'host' if 'host' in spk.lower() else 'guest'

    return qa_pair_count

def create_task_dataset(initial_dataset, output_dir):
    task_dataset = []

    for index, row in initial_dataset.iterrows():
        speakers = eval(row['speaker'])
        qa_length = count_qa_pairs(speakers)

        for k in range(1, qa_length):
            input_text, ground_truth = combine_N_qa_pairs_and_next_question(row, k)
            if ground_truth is not None:
                task_dataset.append([
                    row['id'], 
                    k, 
                    input_text, 
                    ground_truth, 
                    row['interview_goals'], 
                    row['outline_statement'], 
                    row['general_questions']
                ])
    task_dataset_df = pd.DataFrame(task_dataset, columns=[
        'id', 
        'k (num qa_pairs)', 
        'Input (first k qa_pairs)', 
        'Ground Truth (k+1 question)',
        'interview_goals',
        'outline_statement',
        'general_questions'
    ])
    
    output_file_path = os.path.join(output_dir, 'task_dataset.csv')
    os.makedirs(output_dir, exist_ok=True)
    task_dataset_df.to_csv(output_file_path, index=False)
    return task_dataset_df

def downsample_task_dataset(task_dataset_df, max_datapoints=3):
    sampled_df = task_dataset_df.groupby('id').apply(lambda x: x.sample(min(len(x), max_datapoints)))
    sampled_df.reset_index(drop=True, inplace=True)
    return sampled_df

def split_and_downsample_task_datasets(initial_dataset_path, output_base_dir):
    """
    Load the initial dataset, create the task dataset, split it into training and testing sets,
    downsample the sets, and save them to CSV files.
    """

    initial_dataset = pd.read_csv(initial_dataset_path)
    task_dataset_df = create_task_dataset(initial_dataset, output_base_dir)

    # split interviews based on interview id
    interview_ids = task_dataset_df['id'].unique()
    train_ids, test_ids = train_test_split(interview_ids, test_size=0.2, random_state=42)

    train_df = task_dataset_df[task_dataset_df['id'].isin(train_ids)]
    test_df = task_dataset_df[task_dataset_df['id'].isin(test_ids)]

    # downsample
    train_df_downsampled = downsample_task_dataset(train_df)
    test_df_downsampled = downsample_task_dataset(test_df)

    output_dir = os.path.join(output_base_dir, "task_dataset")
    os.makedirs(output_dir, exist_ok=True)
    train_output_file_path = os.path.join(output_dir, 'task_dataset_train.csv')
    test_output_file_path = os.path.join(output_dir, 'task_dataset_test.csv')
    train_df_downsampled.to_csv(train_output_file_path, index=False)
    test_df_downsampled.to_csv(test_output_file_path, index=False)
    return train_df_downsampled, test_df_downsampled

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    initial_dataset_path = os.path.join(base_dir, 'dataset', 'outline', 'transcripts_with_split_outlines_fixed.csv')
    output_base_dir = os.path.join(base_dir, 'dataset', 'task_dataset')
    
    train_df_downsampled, test_df_downsampled = split_and_downsample_task_datasets(initial_dataset_path, output_base_dir)

    print("Train Dataset")
    print(train_df_downsampled.head(10))
    print(f"Shape: {train_df_downsampled.shape[0]}")
    print(train_df_downsampled.columns)

    print("\nTest Dataset")
    print(test_df_downsampled.head(10))
    print(f"Shape: {test_df_downsampled.shape[0]}")
    print(test_df_downsampled.columns)