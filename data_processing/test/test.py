import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def remove_substring_from_column(path, column_name, substring):
    """
    This function removes a specified substring from all the values in a particular column of a DataFrame.
    
    :param df: The DataFrame from which to remove the substring.
    :param column_name: The name of the column in which the substring should be removed.
    :param substring: The substring to remove from the column's values.
    :return: The DataFrame with the substring removed from the specified column.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.read_csv(path)

    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        # Apply the replacement
        df[column_name] = df[column_name].str.replace(substring, '', regex=False)
    else:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
    
    df.to_csv(path, index=False)
    print(f"csv file saved to {path}")
    return df


if __name__ == "__main__":
    baseline_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/baseline/LLM_classified_results.csv"
    CoT_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/CoT/LLM_classified_results.csv"
    outline_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/outline/LLM_classified_results.csv"
    CoT_outline_path = "/project/jonmay_231/spangher/Projects/news-interview-question-generation/output_results/CoT_outline/LLM_classified_results.csv"

    remove_substring_from_column(baseline_path, "Actual_Question_Type", "(MISC) ")
    remove_substring_from_column(baseline_path, "LLM_Question_Type", "(MISC) ")

    remove_substring_from_column(CoT_path, "Actual_Question_Type", "(MISC) ")
    remove_substring_from_column(CoT_path, "LLM_Question_Type", "(MISC) ")

    remove_substring_from_column(outline_path, "Actual_Question_Type", "(MISC) ")
    remove_substring_from_column(outline_path, "LLM_Question_Type", "(MISC) ")

    remove_substring_from_column(CoT_outline_path, "Actual_Question_Type", "(MISC) ")
    remove_substring_from_column(CoT_outline_path, "LLM_Question_Type", "(MISC) ")

    