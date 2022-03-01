import re
import unicodedata

import mne
import numpy as np
import pandas as pd
import scipy.io

def strip_accents(s):
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")

## Broderick-specific utilities

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