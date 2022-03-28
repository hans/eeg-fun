"""
Uses our EEG library to prepare a regression dataset to reproduce the N400
effect in the naturalistic dataset of Frank et al. 2015.

Writes a regression dataset containing the responses / predictors from both
the authors' method and our method. This allows for easy experimentation
downstream.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mfn400.adapters.frank2015 import FrankDatasetAdapter


EPOCH_WINDOW = (-0.1, 0.924)
TEST_WINDOW = (0.3, 0.5)
BASELINE_WINDOW = (None, 0)

N400_ELECTRODES = ["1", "14", "24", "25", "26", "29", "30", "31",
                   "41", "42", "44", "45"]


def sanity_check_and_merge(erp_df: pd.DataFrame, baseline_df: pd.DataFrame,
                           reference_erp_dataset: Path):
    reference_df = pd.read_csv(reference_erp_dataset,
                               index_col=["subject_idx", "sentence_idx", "word_idx"])

    # Double-check indexing.
    for level in ["subject_idx", "sentence_idx"]:
        assert set(reference_df.index.get_level_values(level)) == set(erp_df.index.get_level_values(level)), level
        assert set(erp_df.index.get_level_values(level)) == set(baseline_df.index.get_level_values(level)), level

    comp_df = pd.merge(reference_df, erp_df, left_index=True, right_index=True)
    comp_df = pd.merge(comp_df, baseline_df, left_index=True, right_index=True)

    # Plot given vs derived N400 values.
    sns.scatterplot(data=comp_df.reset_index(), x="value_N400", y="our_N400")
    plt.title("Their N400 value (X) vs ours (Y)")
    plt.savefig("n400_comparison.png")

    return comp_df


def main(args):
    data = FrankDatasetAdapter(args.data_dir, args.stim_df_path)
    erp_df = data.to_erp(EPOCH_WINDOW, TEST_WINDOW, BASELINE_WINDOW,
                         apply_baseline=False)

    # Filter out stimuli of .. unknown origin.
    erp_df = erp_df.loc[erp_df.sentence_idx != -1]

    # Average over N400 electrodes as given in paper.
    our_erp_df = erp_df.reset_index().set_index(["subject_idx", "sentence_idx", "word_idx", "surprisal"]) \
        .loc[:, N400_ELECTRODES].mean(axis=1).rename("our_N400")
    our_erp_df = pd.DataFrame(our_erp_df).reset_index("surprisal")

    our_baseline_df = erp_df.reset_index().set_index(["subject_idx", "sentence_idx", "word_idx"]) \
        .loc[:, [f"{el}_baseline" for el in N400_ELECTRODES]].mean(axis=1).rename("our_baseline")

    merged_df = sanity_check_and_merge(erp_df, args.reference_erp_path)
    merged_df.to_csv(args.out_path)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("data_dir", type=Path)
    p.add_argument("stim_df_path", type=Path)
    p.add_argument("-o", "--out_path", type=Path, required=True)
    p.add_argument("-r", "--reference_erp_path", type=Path, required=True,
                   help=("Path to output of frank2015_erp_premade.py for "
                         "sanity checking derived N400 values."))
