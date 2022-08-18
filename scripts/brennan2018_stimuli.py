# +
"""
Prepare dataframe for Brennan et al. 2018 naturalistic experiment.
Pre-compute surprisals. Merge with existing predictor data.
"""

from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import re
import sys
from typing import *
import unicodedata

import nltk
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import transformers
# -


# +
def preprocess_text(fulltext_path: Path):
    raw_text = fulltext_path.read_text()

    sentences = nltk.tokenize.PunktSentenceTokenizer().tokenize(raw_text)
    sentences = [nltk.tokenize.RegexpTokenizer("[\w']+|[^\w\s]+").tokenize(sent)
                 for sent in sentences]
    tokens_flat = [tok for sent in sentences for tok in sent]

    return sentences, tokens_flat

def strip_accents(s):
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")


# +
punct_re = re.compile(r"[^A-Za-z]")

def process_fulltext_token(t):
    ret = strip_accents(t.lower())
    ret = punct_re.sub("", ret)
    return ret

def align_stimulus_fulltext(stim_df, tokens_flat):
    """
    Prepare an alignment between the eperimental stimuli and the fulltext token
    indices, so that we can provide surprisal predictors.

    Adds a new column `tok_pos` to `stim_df` describing the corresponding
    position for each row in the fulltext. (inplace)
    """

    stop = False

    tok_cursor = 0
    tok_el = process_fulltext_token(tokens_flat[tok_cursor])

    # For each element in surp_df, record the index of the corresponding element
    # in the token sequence or surprisal df.
    tok_pos = []
    for _, row in tqdm(stim_df.iterrows(), total=len(stim_df), desc="Aligning tokens"):
        if stop:
            break

        df_el = punct_re.sub("", row.Word.lower())
        logging.debug("%s :: %s ::", row.Word, df_el)

        # Track how many elements in a reference we have skipped. If this
        # is excessive, we'll quit rather than looping infinitely.
        skip_count = 0
        if stop:
            raise RuntimeError("abort")
            break

        # Find corresponding token in text and append to `tok_pos`.
        try:
            logging.debug("\t/// %r %r", tok_el, df_el)
            while not tok_el.startswith(df_el):
                # Special cases for oddities in the Brennan stim df..
                if tok_el == "is" and df_el == "s":
                    # annotation says "\x1as" which gets stripped
                    break
                elif tok_el == "had" and df_el == "d":
                    # annotation says "\x1ad" which gets stripped
                    break

                tok_cursor += 1
                skip_count += 1
                if skip_count > 20:
                    stop = True
                    break

                tok_el = process_fulltext_token(tokens_flat[tok_cursor])
                logging.debug("\t// %s", tok_el)

            logging.debug("\tMatch %s %s", df_el, tok_el)
            tok_pos.append(tok_cursor)

            # If we matched only a subset of the token, then cut off what we
            # matched and proceed.
            if tok_el != df_el:
                tok_el = tok_el[len(df_el):]
            else:
                tok_cursor += 1
                tok_el = process_fulltext_token(tokens_flat[tok_cursor])
        except IndexError:
            # print("iex", row, tok_cursor, tok_el)
            stop = True
            break

    stim_df["tok_pos"] = tok_pos


# -

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
    for sent_idx, sentence in enumerate(tqdm(sentences, desc="Computing surprisals")):
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

    # Compute frequency-based log2 surprisal.
    total_freq = sum(word_freqs.values())
    word_surps = {word: -np.log2(freq / total_freq)
                  for word, freq in word_freqs.items()}

    stim_df["word_freq"] = stim_df.Word.str.lower().map(word_freqs)
    stim_df["word_freq_surp"] = stim_df.Word.str.lower().map(word_surps)


def main(args):
    stim_df = pd.read_csv(args.stim_path)
    sentences, tokens_flat = preprocess_text(args.fulltext_path)
    
    # Align in-place
    align_stimulus_fulltext(stim_df, tokens_flat)

    if args.surprisals_path:
        surp_mapping = pd.read_csv(args.surprisals_path)
    else:
        surp_mapping = compute_surprisals(stim_df, sentences, model=args.model)

    # Merge stim df with surprisals.
    surp_mapping["token"] = surp_mapping.global_text_tok_idx.map(dict(enumerate(tokens_flat)))
    stim_df = pd.merge(stim_df, surp_mapping.groupby(["global_text_tok_idx"]).surprisal.sum(),
                       how="left", left_on="tok_pos", right_index=True)

    if args.word_freqs_path:
        add_word_freqs(stim_df, args.word_freqs_path)

    stim_df.to_csv(args.out_path, index=False)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("stim_path", type=Path)
    p.add_argument("fulltext_path", type=Path)
    p.add_argument("-o", "--out-path", type=Path, default=sys.stdout)
    p.add_argument("-m", "--model", default="gpt2")

    p.add_argument("--surprisals-path", type=Path)
    p.add_argument("--word-freqs-path", type=Path)

    main(p.parse_args())
