import re
import unicodedata

import mne
import numpy as np
import pandas as pd
import scipy.io


def strip_accents(s):
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")


def run_n400(eeg_paths, data_channels, reference_channels,
             sample_rate,
             stim_df,
             filter_low=1., filter_high=8.,
             epoch_window=(-0.2, 0.5),
             test_window=(0.4, 0.45)):
    """
    Run an N400 analysis for a single subject, whose EEG
    data live in Broderick2018 format at `eeg_paths` (one
    file per run).
    
    Args:
        filter_low: Lower end of band-pass filter in Hz
        filter_high: Higher end of band-pass filter in Hz
        epoch_window: Edges of epoch temporal window
        test_window: Edges of window within each epoch on which to run test 
            relative to zero point
    """
    
    # Load and preprocess EEG data. Returns one concatenated Raw sequence
    # describing all runs.
    raw, run_offsets = load_eeg(eeg_paths, data_channels, reference_channels,
                                sample_rate=sample_rate,
                                filter_low=filter_low,
                                filter_high=filter_high)
    
    # Convert word onset information into MNE event representation.
    events_seq = prepare_events_seq(stim_df, run_offsets)
    
    # Compute epochs based on event representation.
    epoch_tmin, epoch_tmax = epoch_window
    epochs = mne.Epochs(raw, events_seq, preload=True,
                        tmin=epoch_tmin, tmax=epoch_tmax)
    
    # Compute dataframe describing responses at test window.
    # TODO don't enable baselining across run boundaries
    baselined_df = epochs.apply_baseline().crop(*test_window) \
        .to_data_frame(index=["condition", "epoch", "time"])
    # # Merge in item numbers.
    # baselined_df = pd.merge(
    #     baselined_df.reset_index(),
    #     stim_df.reset_index()[["content_word_idx", "item"]],
    #     left_on="epoch",
    #     right_on="content_word_idx"
    # )
    merged_df = pd.concat([
        stim_df.reset_index(),
        baselined_df.groupby(["condition", "epoch"]).mean().reset_index()
    ], axis=1)
    
    return merged_df
    
    
def prepare_events_seq(stim_df, run_offsets, word_event_id=2):
    """
    Prepare an MNE event-matrix representation with one event per
    word onset.
    
    Args:
        stim_df:
        run_offsets: `num_runs` array describing the sample ID at which each 
            run begins in the concatenated sample sequence
    """
    
    events_arr = []
    
    for i, run_offset in enumerate(run_offsets):
        word_offsets = stim_df.loc[i + 1].sample_id + run_offset

        events_arr.append(np.stack([
            word_offsets,
            np.zeros_like(word_offsets),
            word_event_id * np.ones_like(word_offsets)
        ], axis=1))
        
    return np.concatenate(events_arr)
    


## Broderick-specific data utilities

def load_stimuli(path):
    data = scipy.io.loadmat(path)
    
    return pd.DataFrame.from_dict({"word": [el[0][0] for el in data["wordVec"]],
                                   "onset_time": data["onset_time"].flatten(),
                                   "offset_time": data["offset_time"].flatten()})


def load_envelope(paths):
    data = scipy.io.loadmat(path)
    
    return pd.DataFrame.from_dict({"word": [el[0][0] for el in data["wordVec"]],
                                   "onset_time": data["onset_time"].flatten(),
                                   "offset_time": data["offset_time"].flatten()})


info_re = re.compile(r"Subject(\d+)_Run(\d+)\.mat")
def load_eeg(subject_paths, data_channels, reference_channels,
             sample_rate, filter_low=1, filter_high=8):
    """
    Load subject data into a single MNE `RawData` representation.
    
    Runs relevant preprocessing steps as well:
    1. Load and set reference channels
    2. Band-pass filtering
    
    Returns:
        raw_data: An MNE `RawData` representation.
        run_offsets: Integer sample offsets for each of the runs
            for this subject within the raw data, i.e.
                
                [run1_offset, run2_offset, ...]
    """
    channel_names = data_channels + reference_channels
    mne_info = mne.create_info(channel_names, sfreq=sample_rate,
                               ch_types=["eeg"] * len(channel_names))
    
    # Each element of all_data will be (run_id, data)
    # where data is (num_data_channels + num_reference_channels) * num_samples
    all_data = []
    for path in subject_paths:
        subject, run = info_re.findall(path.name)[0]
        data = scipy.io.loadmat(path)
        
        mat = np.concatenate([data["eegData"].T, data["mastoids"].T], axis=0)
        
        # TODO(EEG) is this scaling right?
        mat /= 1e6
        all_data.append((int(run), mat))
    
    all_data = sorted(all_data, key=lambda v: v[0])
    
    raw_ret = mne.io.RawArray(np.concatenate([mat for _, mat in all_data], axis=1),
                              mne_info)
    
    # Set reference.
    # TODO(EEG) is this reference right? just using the average of bilateral channels
    raw_ret = raw_ret.set_eeg_reference(reference_channels)
    
    # Run band-pass filter.
    raw_ret = raw_ret.filter(filter_low, filter_high)
    
    run_offsets = []
    acc = 0
    for _, mat in all_data:
        run_offsets.append(acc)
        acc += mat.shape[1]
        
    return raw_ret, run_offsets