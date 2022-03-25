"""
Prepare dataframe for Frank et al. 2015 naturalistic experiment.
Pre-compute surprisals, control predictors etc.
"""

from argparse import ArgumentParser
from pathlib import Path
import re
from typing import Tuple, List
import unicodedata

import nltk
import numpy as np
import pandas as pd
import scipy.io
import torch
import transformers
from tqdm import tqdm

from mfn400 import transformers_utils



def load_stimuli(path) -> List[List[str]]:
    """
    Load sentence stimuli in original Frank format.
    """
    data = scipy.io.loadmat(path, simplify_cells=True)
    return [list(sentence) for sentence in data["sentences"]]


def add_control_predictors(stim_df, sentences):
    stim_df["word_len"] = stim_df.word.str.len()


def add_word_freqs(stim_df, freqs_data):
    """
    Add word frequency data from `freqs_path` (one word type per line;
    `<word>\\t<freq>`)
    """
    word_freqs = {}
    with open(freqs_data) as f:
        for line in f:
            if line.strip():
                word, freq = line.strip().split("\t")
                word_freqs[word] = int(freq)

    stim_df["word_freq"] = stim_df.word.str.lower().map(word_freqs)


def main(args):
    sentences = load_stimuli(args.stim_path)

    stim_df = pd.concat(
        [pd.DataFrame({"word": sent}) for sent in sentences],
        keys=np.arange(len(sentences)),
        names=["sentence_idx", "word_idx"]
    )

    if args.surprisals_path:
        surp_mapping = pd.read_csv(args.surprisals_path)
    else:
        surp_mapping = transformers_utils.compute_surprisals(sentences, model=args.model)

    # Merge stim df with surprisals.
    stim_df = pd.merge(stim_df.reset_index(),
                       surp_mapping.groupby(["sent_idx", "text_word_idx"]).surprisal.sum(),
                       how="left",
                       left_on=["sentence_idx", "word_idx"],
                       right_index=True)

    # 1-index sentences.
    stim_df = stim_df.reset_index()
    stim_df["sentence_idx"] += 1
    stim_df = stim_df.set_index(["sentence_idx", "word_idx"])

    add_control_predictors(stim_df, sentences)

    if args.word_freqs_path:
        add_word_freqs(stim_df, args.word_freqs_path)

    stim_df.to_csv(args.out_path)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("stim_path", type=Path)
    p.add_argument("-o", "--out-path", type=Path)
    p.add_argument("-m", "--model", default="gpt2")

    p.add_argument("--surprisals-path", type=Path)
    p.add_argument("--word-freqs-path", type=Path)

    main(p.parse_args())
