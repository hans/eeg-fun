"""
This script prepares an ERP analysis regression dataset using the pre-epoched
data provided by Frank et al. 2015.

NB to be consistent with others, indexing is as follows:

- sentence_idx: 1-based
- subject_idx: 1-based
- word_idx: 0-based

Sorry about that. :)
"""

from argparse import ArgumentParser

import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from functools import reduce

from mfn400.adapters.frank2015 import make_signal_df, make_feature_df, make_surprisal_df, make_control_df


def main(args):
    data = scipy.io.loadmat(args.raw_stimuli_path, simplify_cells=True)

    n400_df = pd.merge(make_signal_df(data, "ERP", "value_N400"),
                       make_signal_df(data, "ERPbase", "base_N400"),
                       left_index=True, right_index=True)

    by_participant_features = ["artefact", "reject"]
    results_df = reduce(lambda acc, feature: pd.merge(acc, make_feature_df(data, feature), left_index=True, right_index=True),
                        by_participant_features, n400_df)

    surprisal_df = pd.concat([make_surprisal_df(data, "surp_ngram", "surp_ngram_order", i_offset=2),
                              make_surprisal_df(data, "surp_rnn", "surp_rnn_size", i_offset=1)],
                             axis=1)

    control_features = ["logwordfreq", "wordlength"]
    control_df = reduce(lambda acc, feature: pd.merge(acc, make_control_df(data, feature), left_index=True, right_index=True),
                        control_features[1:], make_control_df(data, control_features[0]))

    merged_df = pd.merge(results_df, surprisal_df, left_index=True, right_index=True)
    merged_df = pd.merge(merged_df, control_df, left_index=True, right_index=True)
    assert len(merged_df) == len(n400_df)

    # Fix indexing.
    merged_df = merged_df.reset_index()
    merged_df["sentence_idx"] += 1
    merged_df["subject_idx"] += 1
    merged_df = merged_df.set_index(["subject_idx", "sentence_idx", "word_idx"]).sort_index()

    merged_df.to_csv(args.outfile)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("raw_stimuli_path", type=Path,
                   default=Path("stimuli_erp.mat"))
    p.add_argument("outfile", type=Path)

    main(p.parse_args())
