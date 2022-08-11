"""
Defines data structures.
"""


from typing import NamedTuple, List, Optional

import torch
from torchtyping import TensorType


# Shape type annotations
n_samples = T = "n_samples"
n_channels = S = "n_channels"
n_words = B = "n_words"
n_phonemes = N_P = "n_phonemes"
n_candidates = N_C = "n_candidates"
n_features_ts = N_F_T = "n_features_ts"
n_features = N_F = "n_features"


class VariableOnsetRegressionDataset(NamedTuple):
    """
    Defines a time series dataset for variable-onset regression.

    The predictors are stored in two groups:

    1. `X_ts`: Time-series predictors, which are sampled at the same rate as `Y`.
    2. `X_variable`: Latent-onset predictors, `batch` many whose onset is to be inferred
       by the model.

    All tensors are padded on the N_P axis on the right to the maximum word length.
    """
    
    sample_rate: int

    phonemes: List[str]
    """
    Phoneme vocabulary.
    """
    
    ## Variable-onset model inputs

    p_word: TensorType[B, N_C, is_log_probability]
    """
    Predictive distribution over expected candidate words at each time step,
    derived from a language model.
    """

    word_lengths: TensorType[B, int]
    """
    Length of ground-truth words in phonemes. Can be used to unpack padded
    ``N_P`` axes.
    """

    candidate_phonemes: TensorType[B, N_C, N_P, int]
    """
    Phoneme ID sequence for each word and alternate candidate set.
    """

    word_onsets: TensorType[B, float]
    """
    Onset of each word in seconds, relative to the start of the sequence.
    """

    phoneme_onsets: TensorType[B, N_P, float]
    """
    Onset of each phoneme within each word in seconds, relative to the start of
    the corresponding word.
    """
    
    ## Response model inputs/targets

    X_ts: TensorType[T, N_F_T, float]

    X_variable: TensorType[B, N_F, float]
    """
    Word-level features whose onset is to be determined by the model.
    """

    Y: TensorType[T, S, float]
    """
    Response data.
    """


# class VariableOnsetRegressionDataset(NamedTuple):
    
#     phoneme: List[str]
#     """
#     Phoneme vocabulary.
#     """
    
#     ## Variable-onset model inputs.
    
#     p_word: TensorType[n_words, n_candidates, float]
#     """
#     Predictive distribution over expected candidate words at each word onset,
#     derived from a language model.
#     """
    
#     ## Response model features.
    
#     Y: TensorType[n_samples, n_channels, float]
    
#     X_control: TensorType[n_samples, "n_f_control", float]
#     """
#     "Control" features are sampled at the same rate as the output
#     signal. These can include things like sound power, speech envelope,
#     acoustic edges, spectral features, etc.
#     """
    
#     X_word: TensorType[n_words, "n_f_word", float]
#     """
#     Word-level features.
#     """
    
#     X_phoneme: TensorType[n_words, n_phonemes, "n_f_phoneme", float]
#     """
#     Phoneme-level features.
#     """