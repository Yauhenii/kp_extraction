# source https://github.com/IBM/KPA_2021_shared_task/blob/main/code/track_1_kp_matching.py

import re

import pandas as pd
import os
from pathlib import Path

from project.data.datasets.processor_base import BaseDatasetProcessor

DATASET_NAME = Path(__file__).resolve().parents[0].name
project_dir = Path(__file__).resolve().parents[4]
gold_data_dir = os.path.abspath(f"{project_dir}/data/raw/ArgKP_2021_shared_task/test_data")
process_data_dir = os.path.abspath(f"{project_dir}/data/processed/ARG_KP_2021")
predictions_data_dir = os.path.abspath(f"{project_dir}/data/predictions")


def load_processed_dataset(processed_path, subset, pred_df: pd.DataFrame = None):
    print("** loading task data:")
    all_df = os.path.join(processed_path, f"{subset}_complete.csv")
    all_df = pd.read_csv(all_df)
    
    if pred_df is not None:
        all_df = all_df[all_df["idx"].isin(pred_df["idx"])]
    return all_df


def load_dataset(gold_data_dir, subset):
    print("** loading task data:")
    arguments_file = os.path.join(gold_data_dir, f"arguments_{subset}.csv")
    key_points_file = os.path.join(gold_data_dir, f"key_points_{subset}.csv")
    labels_file = os.path.join(gold_data_dir, f"labels_{subset}.csv")
    
    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)
    return arguments_df, key_points_df, labels_file_df


def preprocessing(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'', text)
    html = re.compile(r'<.*?>')
    text = html.sub(r'', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)
    return text


def process_data(gold_data_dir, stage, with_undecided_pairs) -> pd.DataFrame:
    arg_df, kp_df, labels_df = load_dataset(gold_data_dir, subset=stage)
    if not with_undecided_pairs:
        merged_df = arg_df.merge(labels_df, on=["arg_id"])
        merged_df = merged_df.merge(kp_df[["key_point", "key_point_id"]], on=["key_point_id"])
        return merged_df
    
    merged_df = arg_df.merge(kp_df, on=["topic", "stance"])
    merged_df = merged_df.merge(
        labels_df,
        how="left",
        on=["arg_id", "key_point_id"]
    )
    # Resolve undecided labels:
    # For strict labels, fill with no match (0).
    merged_df["label_strict"] = merged_df["label"].fillna(0)
    # For relaxed labels, fill with match (1).
    merged_df["label_relaxed"] = merged_df["label"].fillna(1)
    
    # fill not available labels as match (1).
    merged_df["label"] = merged_df["label"].fillna(1)
    
    return merged_df


class ArgKP21Processor(BaseDatasetProcessor):
    
    def process_dataset(self, input_dir: str, subset: str, processed_dir, with_undecided_pairs: bool = True):
        print("** processing task data:")
        
        processed_df = process_data(input_dir, subset, with_undecided_pairs)
        processed_df['argument'] = processed_df['argument'].apply(lambda x: preprocessing(x))
        processed_df['key_point'] = processed_df['key_point'].apply(lambda x: preprocessing(x))
        processed_df['topic'] = processed_df['topic'].apply(lambda x: preprocessing(x))
        processed_df['text'] = processed_df['argument'] + ' ' + processed_df['key_point'] + ' ' + processed_df['topic']
        
        processed_df['sentence1'] = '[CLS] ' + processed_df['argument'] + ' [SEP] ' + processed_df['topic'] + ' [SEP]'
        processed_df['sentence2'] = '[CLS] ' + processed_df['key_point'] + ' [SEP] ' + processed_df['topic'] + ' [SEP]'
        
        # ignore stance and topic for now
        fullname_complete = os.path.join(processed_dir, f"{subset}_complete.csv")
        processed_df.index.name = 'idx'
        processed_df.to_csv(fullname_complete, index=True)
        fullname = os.path.join(processed_dir, f"{subset}.csv")
        processed_df[['text', 'label']].to_csv(fullname, index=False)


if __name__ == "__main__":
    processor = ArgKP21Processor()
    gold_data_dir = os.path.abspath(f"{project_dir}/data/raw/ArgKP_2021_shared_task/kpm_data")
    processor.process_dataset(gold_data_dir, 'train', process_data_dir, with_undecided_pairs=False)
