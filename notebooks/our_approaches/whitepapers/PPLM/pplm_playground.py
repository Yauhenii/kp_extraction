import os

import pandas as pd

from ctg_helper import get_pplm_summary

if __name__ == "__main__":
    params = {
        "pretrained_model": "google/pegasus-xsum",  # "google/pegasus-xsum", #"gpt2-large"
        "length": 50,
        "gamma": 1.5,
        "num_iterations": 3,
        "num_samples": 5,
        "stepsize": 0.02,
        "window_length": 5,
        "kl_scale": 0.01,
        "gm_scale": 0.99,
        "colorama": True,
        "sample": True
    }

    # try with "gpt2-medium"
    # first for full bow

    bow_path = f"{os.path.dirname(os.path.realpath(__file__))}/kpa_dataset/full_bow.txt"
    kpa_test_df = pd.read_csv(f'{os.path.dirname(os.path.realpath(__file__))}/kpa_dataset/processed_kpa.csv',
                              index_col=False)
    promote_test = kpa_test_df.iloc[0]['extracted_kps']
    unpert_gen_list, pert_gen_list = get_pplm_summary(bow_path, promote_test, params)
