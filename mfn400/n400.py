"""
Tools for MNE N400 analysis.
"""

from typing import Tuple

import mne
import pandas as pd



def prepare_erp_df(epochs: mne.Epochs, stim_df: pd.DataFrame,
                   test_window: Tuple[float, float]):
    # Compute dataframe describing responses at test window, relative to
    # epoch_left_edge < t <= 0 baseline mean.
    baselined_df = epochs.copy().apply_baseline().crop(*test_window) \
        .to_data_frame(index=["condition", "epoch", "time"])
    epoch_averages = baselined_df.groupby(["condition", "epoch"]).mean().sort_index(level="epoch").reset_index()
    
    assert len(stim_df) == len(epoch_averages), \
        (f"Mismatched presentation data and epoch data. Something's wrong. "
         f"({len(stim_df)} != {len(epoch_averages)})")
    merged_df = pd.concat([
        stim_df.reset_index(),
        epoch_averages
    ], axis=1)
    
    return merged_df