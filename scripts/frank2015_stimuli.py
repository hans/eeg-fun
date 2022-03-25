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


def compute_surprisals(sentences: List[List[str]], model="gpt2"):
    """
    Compute word-level surprisals and other positional information. Returns a
    dataframe with columns:

        global_text_word_idx: Index into flattened token list of fulltext
        sent_idx:
        text_word_idx: Index into sentence word idx
        tok_idx: Model token idx
        surprisal: log-e surprisal
    """
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model, add_cross_attention=True, is_decoder=True)

    batch_encoding, sentence_surprisals = \
        transformers_utils.get_predictive_outputs(hf_model, hf_tokenizer,
                                                  tqdm(sentences))

    global_tok_cursor = 0

    # Each element is of form (global_text_tok_idx, sent_idx, text_tok_idx, tok_idx, surprisal)
    # `global_text_tok_idx` is an index into `tokens_flat`
    # `sent_idx` is an index into `sentences`
    # `text_tok_idx` is an index into `sentences[sent_idx]`
    surp_mapping = []
    for sent_idx, sentence_surprisals in enumerate(sentence_surprisals):
        sent_tokens = torch.tensor(batch_encoding["input_ids"][sent_idx])

        # get surprisals of expected words
        surps_shifted = sentence_surprisals[:-1, :]
        sent_tokens_shifted = sent_tokens[1:]
        token_surps = surps_shifted[range(surps_shifted.shape[0]), sent_tokens_shifted]

        word_ids = batch_encoding.word_ids(sent_idx)
        for idx, surp in enumerate(token_surps):
            # We are enumerating the shifted list. Get the original token
            # index.
            tok_id = idx + 1
            
            if word_ids[tok_id] is None:
                continue
            elif word_ids[tok_id] >= len(sentences[sent_idx]):
                # Word ID is out-of-bounds. This shouldn't happen. But
                # sometimes it does, because Huggingface tokenizer imputes a
                # different notion of "word" than what is in the pre-tokenized
                # input sentence. See comment in `get_predictive_outputs` for
                # a fix, if you really care about sentence-final surprisals.
                continue
                
            surp_mapping.append((global_tok_cursor + word_ids[tok_id],
                                 sent_idx, word_ids[tok_id], tok_id, surp))

        global_tok_cursor += len(sentences[sent_idx])

    surp_mapping = pd.DataFrame(surp_mapping,
                                columns=["global_text_word_idx", "sent_idx",
                                         "text_word_idx", "tok_idx",
                                         "surprisal"])
    return surp_mapping


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
    surp_mapping = compute_surprisals(sentences, model=args.model)

    stim_df = pd.concat(
        [pd.DataFrame({"word": sent}) for sent in sentences],
        keys=np.arange(len(sentences)),
        names=["sentence_idx", "word_idx"]
    )

    if args.surprisals_path:
        surp_mapping = pd.read_csv(args.surprisals_path)
    else:
        surp_mapping = compute_surprisals(sentences, model=args.model)

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
