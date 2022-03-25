"""
Utilities for working with Huggingface transformers models.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers


def compute_surprisals(sentences: List[List[str]], model="gpt2") -> pd.DataFrame:
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
        get_predictive_outputs(hf_model, hf_tokenizer, sentences)

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


def get_predictive_outputs(model: transformers.PreTrainedModel,
                           tokenizer: transformers.PreTrainedTokenizer,
                           sentences: List[List[str]]) -> Tuple[transformers.BatchEncoding, List[np.ndarray]]:
    """
    Compute the softmax distribution over words for a causal / autoregressive
    language model given the batch of sentences.

    Returns:
        encoding: BatchEncoding
        surprisals: a list of matrices, one per sentence, each `T * vocab_size`
            where `T` is the number of tokens in the corresponding sentence.
    """
    tokenized = tokenizer.batch_encode_plus(
        [" ".join(sentence) for sentence in sentences],
        add_special_tokens=True, return_offsets_mapping=True)
    
    # NB doesn't always handle sentence-final punctuation correctly -- is
    # mapped onto a separate word_id.
    # In principle, we should tokenize with `is_split_into_words=True`.

    ret = []
    for sent_idx, sentence in enumerate(sentences):
        sent_tokens = torch.tensor(tokenized["input_ids"][sent_idx]).unsqueeze(0)

        # Run batches of decoding, accounting for limited sequence input size
        max_len = 512
        past = None
        input_ids = sent_tokens.clone()
        surprisal_outputs = []
        while True:
            with torch.no_grad():
                outputs_b = model(
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

        ret.append(all_surprisals)

    return tokenized, ret
