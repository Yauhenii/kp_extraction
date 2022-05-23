import sys
from typing import List
import pandas as pd
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

from generate_kp import generate_kp

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: List[str], arc: int) -> None:
    """
    argv1: path to csv file with 'arg' and 'topic' columns
    argv2: path to the folder with pretrained model
    argv3: batch size (int)
    """
    arg_topic_df_path = argv[1]
    model_folder_path = argv[2]
    batch_size = int(argv[3])
    arg_topic_df = pd.read_csv(arg_topic_df_path)

    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    model = PegasusForConditionalGeneration.from_pretrained(
        model_folder_path,
        local_files_only=True,
    )

    kp_gen_df = generate_kp(
        arg_topic_df,
        model=model,
        device=device,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    print(kp_gen_df)


if __name__ == "__main__":
    main(sys.argv, len(sys.argv))
