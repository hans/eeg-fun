"""
Uses mfn400 to prepare a regression dataset for N400 ERP analysis
in the naturalistic dataset of Brennan et al. 2018.

Writes a regression dataset containing the responses / predictors.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mfn400.adapters.brennan2018 import BrennanDatasetAdapter


# todo parameterize?
FILTER_WINDOW = (0.5, 20)
EPOCH_WINDOW = (-0.1, 0.924)
TEST_WINDOW = (0.3, 0.5)
BASELINE_WINDOW = (None, 0)

# todo what is right for this montage?
N400_ELECTRODES = ["45", "34", "35", "1"]


def main(args):
    data = BrennanDatasetAdapter(args.data_dir)
    erp_df = data.to_erp(EPOCH_WINDOW, TEST_WINDOW, BASELINE_WINDOW,
                         apply_baseline=False,
                         filter_window=FILTER_WINDOW)

    # Average over N400 electrodes as given in paper.
    surprisal_col = "RNN"
    our_erp_df = erp_df.reset_index().set_index(["subject_idx", "sentence_idx", "word_idx", surprisal_col]) \
        .loc[:, N400_ELECTRODES].mean(axis=1).rename("n400")
    our_erp_df = pd.DataFrame(our_erp_df).reset_index(surprisal_col)

    our_baseline_df = erp_df.reset_index().set_index(["subject_idx", "sentence_idx", "word_idx"]) \
        .loc[:, [f"{el}_baseline" for el in N400_ELECTRODES]].mean(axis=1).rename("baseline")

    merged_df = pd.merge(our_erp_df, our_baseline_df, left_index=True, right_index=True)
    merged_df.to_csv(args.out_path)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("data_dir", type=Path)
    p.add_argument("-o", "--out_path", type=Path, required=True)

    main(p.parse_args())
