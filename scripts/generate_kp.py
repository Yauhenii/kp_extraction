import torch
import pandas as pd
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# generate kp from, arg and topic
def generate_kp(
    arg_topic_df: pd.DataFrame,
    model: PegasusForConditionalGeneration,
    tokenizer: PegasusTokenizer,
    device: str,
    use_topic: bool = False,
    batch_size: int = 16,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    arg_topic_df: df with 'arg' and 'topic' columns
    model: pretrained model
    tokenizer: Pegasus Tokenizer
    device: 'cpu'/'gpu'
    use_topic: use arg+topic to predict kp
    batch_size: batch size (int)
    verbose: print intermediate steps
    ________
    return: df with generated kp,arg, and topic(if available)
    """
    feature_df = arg_topic_df.copy()
    if use_topic:
        feature_df["feature"] = feature_df["arg"] + feature_df["topic"]
    else:
        feature_df["feature"] = feature_df["arg"]

    features = list(feature_df["feature"])
    features_count = len(features)
    batch_count = features_count // batch_size + features_count % batch_size

    curr_batch_size = 0
    targets = []

    with torch.no_grad():
        for batch_id in range(batch_count):
            if (batch_id + 1) * batch_size > features_count:
                curr_batch_size = features_count - batch_id * batch_size
            else:
                curr_batch_size = batch_size

            if verbose:
                print(f"batch_id: {batch_id}")
                print(f"curr_batch_size: {curr_batch_size}")
                print("Tokenizing...")
            tokenized_features = tokenizer(
                features[
                    batch_id * batch_size : batch_id * batch_size + curr_batch_size
                ],
                truncation=True,
                padding="longest",
                return_tensors="pt",
            ).to(device)
            if verbose:
                print("Generating...")
            tokenized_targets = model.generate(**tokenized_features, num_beams=6)
            if verbose:
                print("Decoding...")
            targets += tokenizer.batch_decode(
                tokenized_targets,
                skip_special_tokens=True,
            )

    feature_df["kp_gen"] = targets
    return feature_df
