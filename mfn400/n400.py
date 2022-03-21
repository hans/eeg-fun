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
    merged_df = pd.concat([
        stim_df.reset_index(),
        baselined_df.groupby(["condition", "epoch"]).mean().reset_index()
    ], axis=1)
    
    return merged_df