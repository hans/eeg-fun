#!/usr/bin/env python
# coding: utf-8

# This notebook synthesizes and saves EEG-like data in a CDR-friendly format.
# 
# We are testing here whether an N400-like response is recoverable by CDR(NN).

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


def simple_peak(x):
    """Function which rapidly peaks and returns to baseline"""
    return st.gamma.pdf(x * 8, 2.5, 0.1)


def sample_item(n_words=20, word_delay_range=(0.3, 1),
                response_window=(0.0, 1.5),  # time window over which word triggers signal response
                sample_rate=128, n400_surprisal_coef=-1,
                surprisal_mean=2., surprisal_sigma=0.2):
    # sample unitary response
    peak_xs = np.linspace(*response_window, num=int(response_window[1] * sample_rate))
    peak_ys = simple_peak(peak_xs)

    # sample stimulus times+surprisals
    stim_delays = np.random.uniform(*word_delay_range, size=n_words)
    stim_onsets = np.cumsum(stim_delays)
    # hack: align times to sample rate to make this easier
    stim_onsets = np.round(stim_onsets * sample_rate) / sample_rate
    surprisals = np.random.lognormal(surprisal_mean, surprisal_sigma, size=n_words)

    max_time = stim_onsets.max() + response_window[1]
    all_times = np.linspace(0, max_time, num=int(np.ceil(max_time * sample_rate)))
    
    signal = np.zeros_like(all_times)
    for onset, surprisal in zip(stim_onsets, surprisals):
        sample_idx = int(onset * sample_rate)  # guaranteed to be round because of hack.
        signal[sample_idx:sample_idx + len(peak_xs)] += peak_ys * n400_surprisal_coef * surprisal
        
    # build return X, y dataframes.
    X = pd.DataFrame({"time": stim_onsets, "surprisal": surprisals})
    y = pd.DataFrame({"time": all_times, "signal": signal})
    return X, y


def sample_dataset(size, n_word_range=(10, 25), **item_kwargs):
    ret_X, ret_y = [], []
    item_sizes = np.random.randint(*n_word_range, size=size)
    for size in item_sizes:
        X, y = sample_item(n_words=size, **item_kwargs)
        ret_X.append(X)
        ret_y.append(y)
        
    X = pd.concat(ret_X, names=["item", "word_idx"], keys=np.arange(len(ret_X)))
    y = pd.concat(ret_y, names=["item", "sample_idx"], keys=np.arange(len(ret_y)))
    return X, y


def main(args):
    X, y = sample_dataset(args.n_items, sample_rate=args.sample_rate,
                          surprisal_mean=args.surprisal_mean,
                          surprisal_sigma=args.surprisal_sigma,
                          n400_surprisal_coef=args.n400_coef)
    
    X.to_csv(args.outdir / "X.txt", sep=" ")
    y.to_csv(args.outdir / "y.txt", sep=" ")
    
    
if __name__ == "__main__":
    p = ArgumentParser()
    
    p.add_argument("outdir", type=Path)
    p.add_argument("-n", "--n_items", type=int, default=100)
    p.add_argument("--sample_rate", type=int, default=128)
    
    # mean, sigma params for a log-normal surprisal distribution
    p.add_argument("--surprisal_mean", type=float, default=2)
    p.add_argument("--surprisal_sigma", type=float, default=0.2)
    
    # coefficient linking surprisal value to n400 amplitude
    p.add_argument("--n400_coef", type=float, default=-1.)
    
    main(p.parse_args())