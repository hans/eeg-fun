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


# Frank data contains pre-computed response data for a variety of ERPs. We're
# just extracting N400 data here.
DATA_COMPONENTS = ["ELAN", "LAN", "N400", "EPNP", "P600", "PNP"]
N400_COMPONENT_IDX = DATA_COMPONENTS.index("N400")


def make_signal_df(data, key, target_key=None):
    """
    Build dataframe describing ERP signal from Frank formatted struct.
    """
    target_key = target_key or key
    return pd.concat(
        {sentence_idx: pd.DataFrame(mat[:, :, N400_COMPONENT_IDX],
                                    index=pd.RangeIndex(mat.shape[0], name="word_idx")) \
            .reset_index().melt(id_vars=["word_idx"], var_name="subject_idx",
                                value_name=target_key)
            for sentence_idx, mat in enumerate(data[key])},
        names=["sentence_idx", "idx"]) \
        .reset_index().drop(columns=["idx"]) \
        .set_index(["subject_idx", "sentence_idx", "word_idx"])


def make_feature_df(data, key, target_key=None, final_axis_name="subject_idx"):
    target_key = target_key or key
    return pd.concat(
        {sentence_idx: pd.DataFrame(mat,
                                    index=pd.RangeIndex(mat.shape[0], name="word_idx")) \
                                    .reset_index().melt(
                                        id_vars=["word_idx"],
                                        var_name=final_axis_name,
                                        value_name=target_key)
         for sentence_idx, mat in enumerate(data[key])},
        names=["sentence_idx", "idx"]) \
        .reset_index().drop(columns=["idx"]) \
        .set_index(["subject_idx", "sentence_idx", "word_idx"])


def make_surprisal_df(data, key, target_key=None, i_offset=0):
    target_key = target_key or key
    return pd.concat(
        {sentence_idx: pd.DataFrame(mat, index=pd.RangeIndex(mat.shape[0], name="word_idx"),
                                    columns=[f"{target_key}_{i_offset+idx}" for idx in range(mat.shape[1])])
         for sentence_idx, mat in enumerate(data[key])},
        names=["sentence_idx", "word_idx"])


def make_control_df(data, key):
    return pd.concat([pd.DataFrame({key: wf}, index=pd.RangeIndex(len(wf), name="word_idx"))
                      for idx, wf in enumerate(data[key])],
                     names=["sentence_idx"], keys=np.arange(len(data[key])))


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
