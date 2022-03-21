"""
Prepare dataframe for Broderick et al. 2018 naturalistic experiment.
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


def strip_accents(s):
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")


def load_stimuli(path) -> pd.DataFrame:
    """
    Load stimuli in original Broderick format.
    """
    data = scipy.io.loadmat(path)
    df = pd.DataFrame.from_dict({"word": [el[0][0] for el in data["wordVec"]],
                                 "onset_time": data["onset_time"].flatten(),
                                 "offset_time": data["offset_time"].flatten()})

    return df


def preprocess_text(fulltext_path: Path) -> Tuple[List[List[str]], List[str]]:
    with fulltext_path.open() as f:
        text = f.read()

    sentences = nltk.tokenize.PunktSentenceTokenizer().tokenize(text)
    sentences = [nltk.tokenize.RegexpTokenizer("[\w']+|[^\w\s]+").tokenize(sent)
                 for sent in sentences]
    tokens_flat = [tok for sent in sentences for tok in sent]

    return sentences, tokens_flat


def align_stimulus_fulltext(stim_df, tokens_flat):
    """
    Prepare an alignment between the eperimental stimuli and the fulltext token
    indices, so that we can provide surprisal predictors.

    Adds a new column `tok_pos` to `stim_df` describing the corresponding
    position for each row in the fulltext. (inplace)
    """

    punct_re = re.compile(r"[^A-Za-z]")

    stop = False

    tok_cursor = 0
    tok_el = tokens_flat[tok_cursor]

    # For each element in surp_df, record the index of the corresponding element
    # in the token sequence or surprisal df.
    tok_pos = []
    for item, rows in tqdm(stim_df.groupby("item")):
        if stop:
            break

        # print("==========", item)
        for idx, row in rows.iterrows():
            # print(row.word, "::")

            # Track how many elements in a reference we have skipped. If this
            # is excessive, we'll quit rather than looping infinitely.
            skip_count = 0
            if stop:
                break

            # Find corresponding token in text and append to `tok_pos`.
            try:
                tok_el = punct_re.sub("", tok_el)
                while not tok_el.startswith(row.word.lower()):
                    tok_cursor += 1
                    skip_count += 1
                    if skip_count > 20:
                        stop = True
                        break

                    tok_el = strip_accents(tokens_flat[tok_cursor].lower())
                    # print("\t//", element)

                # print("\tMatch", row.word, element)
                tok_pos.append(tok_cursor)

                # If we matched only a subset of the token, then cut off what we
                # matched and proceed.
                if tok_el != row.word.lower():
                    tok_el = tok_el[len(row.word):]
            except IndexError:
                # print("iex", row, tok_cursor, tok_el)
                stop = True
                break

    stim_df["tok_pos"] = tok_pos


def compute_surprisals(stim_df: pd.DataFrame, sentences: List[List[str]],
                       model="gpt2"):
    """
    Compute token-level surprisals and other positional information. Returns a
    dataframe with columns:

        global_text_tok_idx: Index into flattened token list of fulltext
        sent_idx:
        text_tok_idx: Index into sentence word idx
        tok_idx: Model token idx
        surprisal: log-e surprisal
    """
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model, add_cross_attention=True, is_decoder=True)

    tokenized = hf_tokenizer.batch_encode_plus(
        [" ".join(sentence) for sentence in sentences],
        add_special_tokens=True, return_offsets_mapping=True)

    global_tok_cursor = 0

    # Each element is of form (global_text_tok_idx, sent_idx, text_tok_idx, tok_idx, surprisal)
    # `global_text_tok_idx` is an index into `tokens_flat`
    # `sent_idx` is an index into `sentences`
    # `text_tok_idx` is an index into `sentences[sent_idx]`
    surp_mapping = []
    for sent_idx, sentence in enumerate(tqdm(sentences)):
        sent_tokens = torch.tensor(tokenized["input_ids"][sent_idx]).unsqueeze(0)

        # Run batches of decoding, accounting for limited sequence input size
        max_len = 512
        past = None
        input_ids = sent_tokens.clone()
        surprisal_outputs = []
        while True:
            with torch.no_grad():
                outputs_b = hf_model(
                    input_ids=input_ids[:, :max_len],
                    past_key_values=past,
                    return_dict=True)

            # at most max_len * vocab_size
            surprisal_outputs.append(-outputs_b["logits"].log_softmax(dim=2).squeeze(0).numpy())

            past = outputs_b["past_key_values"]

            if input_ids.shape[1] <= max_len:
                # Done.
                break
            else:
                input_ids = input_ids[:, max_len:]

        # T * vocab_size
        all_surprisals = np.concatenate(surprisal_outputs, axis=0)

        # get surprisals of expected words
        surps_shifted = all_surprisals[:-1, :]
        sent_tokens_shifted = sent_tokens.squeeze()[1:]
        token_surps = surps_shifted[range(surps_shifted.shape[0]), sent_tokens_shifted]

        word_ids = tokenized.word_ids(sent_idx)
        for tok_id, surp in enumerate(token_surps):
            if word_ids[tok_id] is None:
                continue
            surp_mapping.append((global_tok_cursor + word_ids[tok_id],
                                 sent_idx, word_ids[tok_id], tok_id, surp))

        global_tok_cursor += len(sentence)

    surp_mapping = pd.DataFrame(surp_mapping,
                                columns=["global_text_tok_idx", "sent_idx",
                                         "text_tok_idx", "tok_idx",
                                         "surprisal"])
    return surp_mapping


def add_control_predictors(stim_df, sentences):
    # Compute sentence idx + token position within sentence
    sentence_idxs = [sentence_idx
                     for sentence_idx, sentence in enumerate(sentences)
                     for tok in sentence]
    sentence_token_idxs = [token_idx
                           for sentence in sentences
                           for token_idx, token in enumerate(sentence)]

    stim_df["sentence_idx"] = stim_df.tok_pos.map(dict(enumerate(sentence_idxs)))
    stim_df["sentence_token_idx"] = stim_df.tok_pos.map(dict(enumerate(sentence_token_idxs)))

    stim_df["sentence_idx_within_item"] = \
        stim_df.sentence_idx - stim_df.groupby("item").sentence_idx.min()

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
    stim_paths = list(args.stim_dir.glob("*.mat"))
    stim_df = pd.concat([load_stimuli(path) for path in stim_paths],
                        keys=[int(p.stem.replace("Run", "")) for p in stim_paths],
                        names=["item", "content_word_idx"]).sort_index()

    sentences, tokens_flat = preprocess_text(args.fulltext_path)

    align_stimulus_fulltext(stim_df, tokens_flat)

    if args.surprisals_path:
        surp_mapping = pd.read_csv(args.surprisals_path)
    else:
        surp_mapping = compute_surprisals(stim_df, sentences, model=args.model)

    # Merge stim df with surprisals.
    surp_mapping["token"] = surp_mapping.global_text_tok_idx.map(dict(enumerate(tokens_flat)))
    stim_df = pd.merge(stim_df, surp_mapping.groupby(["global_text_tok_idx"]).surprisal.sum(),
                   how="left", left_on="tok_pos", right_index=True)

    add_control_predictors(stim_df, sentences)

    if args.word_freqs_path:
        add_word_freqs(stim_df, args.word_freqs_path)

    stim_df.to_csv(args.out_path)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("stim_dir", type=Path)
    p.add_argument("fulltext-path", type=Path)
    p.add_argument("-o", "--out-path", type=Path)
    p.add_argument("-m", "--model", default="gpt2")

    p.add_argument("--surprisals-path", type=Path)
    p.add_argument("--word-freqs-path", type=Path)
