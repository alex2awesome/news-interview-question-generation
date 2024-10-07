# generate_info_items_dict.py

import os
import sys
import json
import re
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions import find_project_root

def clean_info_items(info_items_str):
    return info_items_str.replace('*', '')

def process_info_items(info_items_str):
    items = info_items_str.strip().split('\n')
    info_dict = {}
    current_key = None
    current_value = []

    for item in items:
        item = item.strip()
        match = re.match(r'-?\s*Information item #?(\d+):?\s*(.*)', item, re.IGNORECASE)
        if match:
            if current_key:
                info_dict[current_key] = ' '.join(current_value).strip()
            current_key = f"Information item #{match.group(1)}"
            current_value = [match.group(2).strip()]
        elif current_key:
            current_value.append(item)

    if current_key:
        info_dict[current_key] = ' '.join(current_value).strip()

    return info_dict

if __name__ == "__main__": 
    print(f"generate_info_items_dict.py is Running...\n")
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(current_path, 'news-interview-question-generation')
    dataset_path = os.path.join(project_root, "output_results/game_sim/info_items/final_df_with_info_items.csv")
    
    output_dir = os.path.join(project_root, "output_results/game_sim/info_items_dict")
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(dataset_path)
    print(df)

    df['info_items'] = df['info_items'].apply(clean_info_items)
    df['info_items_dict'] = df['info_items'].apply(process_info_items)
    df['info_items_dict'] = df['info_items_dict'].apply(json.dumps)

    df.to_csv(os.path.join(output_dir, 'final_df_with_info_items_dict.csv'), index=False)
    print(df)

    print(f"\nFinished Running!")