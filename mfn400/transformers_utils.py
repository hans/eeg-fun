"""
Utilities for working with Huggingface transformers models.
"""

from typing import List, Tuple

import numpy as np
import torch
import transformers


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
