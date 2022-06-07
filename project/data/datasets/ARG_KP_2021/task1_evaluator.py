# source https://github.com/IBM/KPA_2021_shared_task/blob/main/code/track_1_kp_matching.py
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import torch.cuda

from numpy import mean
from pandas import DataFrame
from transformers import AutoTokenizer

from project.data.datasets.ARG_KP_2021.processor import load_dataset, load_processed_dataset

import pandas as pd
from sklearn.metrics import average_precision_score

from datasets import Dataset

from project.data.datasets.validator_base import BaseDatasetEvaluator
from project.data.nlp_data_module import NLPDataModule
from project.models.common.base_nlp_model import BaseNLPModel
from project.models.common.nlp_configs import HFTransformerDataConfig

project_dir = Path(__file__).resolve().parents[4]
test_data_path = os.path.abspath(f"{project_dir}/data/processed/ARG_KP_2021/test_complete.csv")
train_data_path = os.path.abspath(f"{project_dir}/data/processed/ARG_KP_2021/all_complete.csv")


def evaluate_predictions(merged_df: DataFrame):
    """
    Calculate mean average precision scores
    for relaxed and strict ground truth labels
    and print evaluation scores to console.
    """
    map_strict = calculate_mean_average_precision(merged_df, "label_strict")
    map_relaxed = calculate_mean_average_precision(merged_df, "label_relaxed")
    print(f"mAP strict= {map_strict} ; mAP relaxed = {map_relaxed}")
    return map_strict, map_relaxed


def calculate_average_precision(df: DataFrame, label_column: str,
                                top_percentile: float = 0.5):
    """
    Calculate average precision score for top-scores in the data frame
    with respect to relaxed or strict ground truth labels.
    """
    top = int(len(df) * top_percentile)
    df = df.sort_values("score", ascending=False).head(top)
    # If labels don't contain any match, define average precision as zero.
    # Otherwise, calculating precision would fail because of division by zero.
    if (df[label_column] == 0).all():
        return 0
    return average_precision_score(
        y_true=df[label_column],
        y_score=df["score"]
    )


def calculate_mean_average_precision(df: DataFrame, label_column: str):
    """
    Calculate mean average precision score for top-scores
    with respect to relaxed or strict ground truth labels
    across argument key point pairs of same topic and stance.
    """
    precisions = [
        calculate_average_precision(group, label_column)
        for _, group in df.groupby(["topic", "stance"])
    ]
    return mean(precisions)


def load_kpm_data(
        gold_data_dir: Path,
        subset: str
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load arguments, key points, and ground truth labels from
    the data directory.
    """
    
    arguments_file = gold_data_dir / f"arguments_{subset}.csv"
    key_points_file = gold_data_dir / f"key_points_{subset}.csv"
    labels_file = gold_data_dir / f"labels_{subset}.csv"
    
    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)
    
    return arguments_df, key_points_df, labels_file_df


def load_predictions(predictions_file: Path) -> DataFrame:
    """
    Load data frame with argument key point matches and scores.
    """
    args: List[str] = []
    kps: List[str] = []
    scores: List[float] = []
    with predictions_file.open("rb") as file_in:
        predictions = pickle.load(file_in)
    for arg_id, kp_scores in predictions.items():
        for kp_id, score in kp_scores.items():
            args.append(arg_id)
            kps.append(kp_id)
            scores.append(score)
    print(f"Loaded {len(args)} predictions "
          f"for {len(predictions.items())} arguments.")
    return DataFrame({
        "arg_id": args,
        "key_point_id": kps,
        "score": scores
    })


def merge_labels_with_predictions(
        predictions_df: DataFrame,
        processed_dataset: DataFrame = None,
        arg_df: DataFrame = None,
        kp_df: DataFrame = None,
        labels_df: DataFrame = None
) -> DataFrame:
    """
    Merge ground truth labels and predicted match scores
    for all argument key point pairs.
    Missing values are filled in as described in the task description.
    """
    if processed_dataset is None:
        # Remove text columns. These are not needed for evaluation.
        arg_df = arg_df.drop(columns=["argument"])
        kp_df = kp_df.drop(columns=["key_point"])
        
        # Create a data frame with all argument key point pairs
        # of same topic and stance.
        merged_df: DataFrame = arg_df.merge(kp_df, on=["topic", "stance"])
        
        # Add ground truth labels.
        # Note that we left-join here, because some pairs have undecided labels,
        # i.e., >15% annotators yet <60% of them marked the pair as a match
        # (as detailed in Bar-Haim et al., ACL-2020).
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
    else:
        merged_df = processed_dataset
    
    # Add participant predictions.
    # Note that we left-join here, because some pairs
    # might not have been predicted by the participant.
    merged_df = merged_df.merge(
        predictions_df,
        how="left",
        on=["arg_id", "key_point_id"]
    )
    # Warn if some pairs were not predicted.
    if (merged_df["score"].isna()).any():
        print(
            "Warning: Not all argument key point pairs were predicted. "
            "Missing predictions will be treated as no match."
        )
    # Fill unpredicted labels, treat them as no match (0).
    merged_df["score"] = merged_df["score"].fillna(0)
    # Select best-scored key point per argument.
    # Shuffle (with seed 42) before sorting to ensure
    # that a random key point is selected in case of a tie.
    merged_df = (merged_df.groupby(by="arg_id")
                 .apply(lambda df: df
                        .sample(frac=1, random_state=42)
                        .sort_values(by="score", ascending=False)
                        .head(1))
                 .reset_index(drop=True))
    return merged_df


def create_predictions(model, dataset, raw_data, batch_size, prediction_file_path):
    preds_dict = {}
    
    args = []
    kps = []
    idxs = []
    scores = []
    
    def add_preds(examples):
        similarity, _, output = model._forward(examples, predict_with_gpu=True)
        for i, idx in enumerate(examples["idx"]):
            row = raw_data.iloc[idx.item()]
            if output is not None:
                score = output[i][1].item()
            else:
                score = similarity[i].item()
            if row["arg_id"] not in preds_dict:
                preds_dict[row["arg_id"]] = {}
            args.append(row["arg_id"])
            kps.append(row["key_point_id"])
            idxs.append(idx.item())
            scores.append(score)
            preds_dict[row["arg_id"]][row["key_point_id"]] = score
    
    dataset.map(lambda examples: add_preds(examples), batched=True, batch_size=batch_size)
    
    with open(f"{prediction_file_path}/prediction_file.txt", "wb") as handle:
        pickle.dump(preds_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    preds_df = DataFrame({
        "arg_id": args,
        "key_point_id": kps,
        "idx": idxs,
        "score": scores
    })
    
    preds_df.to_csv(f"{prediction_file_path}/prediction_file.csv")
    
    return preds_df


class ArgKP21Task1Evaluator(BaseDatasetEvaluator):
    
    def __init__(self):
        self.test_df = pd.read_csv(test_data_path, index_col=None)
        self.train_df = pd.read_csv(train_data_path, index_col=None)
        self.test_df.set_index('idx')
        self.train_df.set_index('idx')
        self.predictions_file = os.path.abspath(f"{project_dir}/data/predictions")
    
    def evaluate(self, model: BaseNLPModel, test_dataset: Dataset, stage: str = 'test', batch_size: int = 32):
        """
         predictions expected format:
         {
           "arg_15_0":{
              "kp_15_0":0.8282181024551392,
              "kp_15_2":0.9438725709915161
           },
           "arg_15_1":{
              "kp_15_0":0.9994438290596008,
              "kp_15_2":0
           }
        }
         """
        if stage == 'test':
            full_data = self.test_df
        else:
            full_data = self.train_df
        
        predictions_df = create_predictions(model, test_dataset, full_data, batch_size, self.predictions_file)
        
        model.to('cuda')
        
        # predictions_df = load_predictions(Path(self.predictions_file))
        processed_dataset = load_processed_dataset(f"{project_dir}/data/processed/ARG_KP_2021",
                                                   subset='test' if stage == 'test' else 'all',
                                                   pred_df=predictions_df)
        merged_df = merge_labels_with_predictions(
            predictions_df,
            processed_dataset
        )
        return evaluate_predictions(merged_df)


def main_with_model():
    torch.cuda.empty_cache()
    from project.models.baselines.classification_model import ClassificationModel
    
    arg_2021_val = ArgKP21Task1Evaluator()
    project_dir = Path(__file__).resolve().parents[4]
    model_checkpoint_path = os.path.abspath(
        f"{project_dir}/project/models/baselines/classification/outputs/2021-12-28/01-04-58/.neptune/None/version_None/checkpoints/epoch=4-step=13554.ckpt")
    model = ClassificationModel.load_from_checkpoint(checkpoint_path=model_checkpoint_path)
    data_cfg = HFTransformerDataConfig(**{
        "batch_size": 32,
        "num_workers": 4,
        "train_file": test_data_path,
        "test_file": test_data_path
    })
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model.model.config._name_or_path,
                                              use_fast=True)
    datamodule = NLPDataModule(cfg=data_cfg, tokenizer=tokenizer)
    datamodule.setup("test")
    test_dataset = datamodule.ds["test"]
    arg_2021_val.evaluate(model, test_dataset, 'test')


def main_with_prediction_file():
    predictions_file = os.path.abspath(f"{project_dir}/data/predictions/prediction_file.csv")
    predictions_df = pd.read_csv(Path(predictions_file), index_col=False)
    processed_dataset = load_processed_dataset(f"{project_dir}/data/processed/ARG_KP_2021", subset='all',
                                               pred_df=predictions_df)
    merged_df = merge_labels_with_predictions(
        predictions_df,
        processed_dataset
    )
    evaluate_predictions(merged_df)


if __name__ == '__main__':
    main_with_prediction_file()
