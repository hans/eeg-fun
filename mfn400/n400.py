"""
Tools for MNE N400 analysis.
"""

import logging
from typing import Tuple

import mne
import numpy as np
import pandas as pd



def get_baseline_values(epochs: mne.Epochs, baseline=(None, 0)):
    """
    Extracted from `mne.baseline.rescale`.
    
    Returns a wide dataframe with one row per epoch, one column per channel.
    NB does not observe channel picks.
    """
    bmin, bmax = baseline
    times = epochs.times
    
    if bmin is None:
        imin = 0
    else:
        imin = np.where(times >= bmin)[0]
        if len(imin) == 0:
            raise ValueError('bmin is too large (%s), it exceeds the largest '
                             'time value' % (bmin,))
        imin = int(imin[0])
    if bmax is None:
        imax = len(times)
    else:
        imax = np.where(times <= bmax)[0]
        if len(imax) == 0:
            raise ValueError('bmax is too small (%s), it is smaller than the '
                             'smallest time value' % (bmax,))
        imax = int(imax[-1]) + 1
    if imin >= imax:
        raise ValueError('Bad rescaling slice (%s:%s) from time values %s, %s'
                         % (imin, imax, bmin, bmax))

    # NB, scales just as to_data_frame would.
    ret = epochs.get_data()[..., imin:imax].mean(axis=-1) * 1e6
    return pd.DataFrame(
        ret,
        index=pd.Index(epochs.selection, name="epoch"),
        columns=epochs.ch_names
    )


def prepare_erp_df(epochs: mne.Epochs, stim_df: pd.DataFrame,
                   test_window: Tuple[float, float],
                   baseline=(None, 0), apply_baseline=True):
    """
    Compute dataframe describing responses at test window, relative to
    baseline mean specified by `baseline`.
    
    Args:
        baseline: MNE baseline spec. See e.g. `mne.Epochs.apply_baseline`.
        apply_baseline: If `True`, values returned will be baselined by
            mean value in baseline window. If `False`, these baseline
            values will be returned as separate columns.
    """
    if not apply_baseline and epochs.baseline is not None:
        raise ValueError("Cannot fetch raw values because this `Epochs` "
                         "object was instantiated with a non-None baseline: "
                        f"{epochs.baseline}")

    epochs = epochs.copy()
    if apply_baseline:
        epochs = epochs.apply_baseline(baseline)
    else:
        baseline_df = get_baseline_values(epochs, baseline=baseline)
        
    epochs_df = epochs.crop(*test_window) \
        .to_data_frame(index=["condition", "epoch", "time"])
    epoch_averages = epochs_df.groupby(["condition", "epoch"]).mean().sort_index(level="epoch")
    
    if not apply_baseline:
        # Merge in baseline information.
        epoch_averages = pd.merge(epoch_averages, baseline_df,
                                  left_index=True, right_index=True,
                                  suffixes=("", "_baseline"))
        
    epoch_averages = epoch_averages.reset_index()
    
    # If epochs were dropped in preprocessing, drop the corresponding
    # stimulus rows.
    dropped_epochs = [idx for idx, log in enumerate(epochs.drop_log)
                      if len(log) > 0]
    if len(dropped_epochs) > 0:
        logging.warning(f"{len(dropped_epochs)} dropped epochs; will drop "
                         "corresponding stimulus rows.")
        mask = np.repeat(True, len(stim_df))
        mask[dropped_epochs] = False
        stim_df = stim_df.loc[mask]
    
    assert len(stim_df) == len(epoch_averages), \
        (f"Mismatched presentation data and epoch data. Something's wrong. "
         f"({len(stim_df)} != {len(epoch_averages)})")
    merged_df = pd.concat([
        stim_df.reset_index(),
        epoch_averages
    ], axis=1)
    
    return merged_df