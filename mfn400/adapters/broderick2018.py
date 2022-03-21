import itertools
from pathlib import Path
import re
from typing import Optional, Tuple, List, Dict

import mne
import numpy as np
import pandas as pd
import scipy.io

from mfn400.adapters import MNEDatasetAdapter, DatasetAdapter


info_re = re.compile(r"Subject(\d+)_Run(\d+)\.mat")


class BroderickDatasetAdapter(MNEDatasetAdapter):
    """
    Broderick et al. 2018 naturalistic N400 dataset.

    Works with stimulus data preprocessed by script
    `scripts/broderick2018_stimuli.py`
    """

    name = "broderick2018"

    # acquisition parameters
    data_channels = [f"V{x}" for x in range(1, 129)]
    # two mastoid references
    reference_channels = ["M1", "M2"]
    sample_rate = 128

    def __init__(self, eeg_dir, stim_path):
        self._prepare_paths(eeg_dir)

        self._stim_df = pd.read_csv(stim_path)
        self._load_mne()

    def _prepare_paths(self, eeg_dir,):
        eeg_paths = itertools.groupby(sorted(eeg_dir.glob("**/*.mat")),
                                      lambda p: info_re.match(p.name).group(1))
        self._eeg_paths = {k: list(v) for k, v in eeg_paths}

    def _load_mne(self):
        """
        Load MNE continuous representation.
        """
        raw_data, run_offsets = {}, {}
        for subject_id, run_paths in self._eeg_paths.items():
            raw_data[subject_id], run_offsets[subject_id] = \
                self._load_mne_single_subject(subject_id, run_paths)

        self._raw_data = raw_data
        self._run_offsets = run_offsets

    def _load_mne_single_subject(self, subject_id, run_paths) -> Tuple[mne.RawData, List[int]]:
        channel_names = self.data_channels + self.reference_channels
        mne_info = mne.create_info(channel_names, sfreq=self.sample_rate,
                                   ch_types=["eeg"] * len(channel_names))

        # Each element of all_data will be (run_id, data)
        # where data is (num_data_channels + num_reference_channels) * num_samples
        all_data = []
        for run_id, path in run_paths.items():
            data = scipy.io.loadmat(path)

            mat = np.concatenate([data["eegData"].T, data["mastoids"].T], axis=0) \
                .astype(np.float32)

            # TODO(EEG) is this scaling right?
            mat /= 1e6
            all_data.append((int(run_id), mat))

        all_data = sorted(all_data, key=lambda v: v[0])

        raw_data = mne.concatenate_raws(
            [mne.io.RawArray(mat, mne_info) for _, mat in all_data]
        )

        run_offsets = []
        acc = 0
        for _, mat in all_data:
            run_offsets.append(acc)
            acc += mat.shape[1]

        return raw_data, run_offsets

    def _preprocess_mne(self, raw_data: mne.RawData,
                        filter_window: Tuple[float, float]) -> mne.RawData:
        # Set reference.
        # TODO(EEG) is this reference right? just using the average of
        # bilateral channels
        raw_data = raw_data.set_eeg_reference(self.reference_channels)

        # Run band-pass filter.
        raw_data = raw_data.filter(*filter_window)

        return raw_data

    @property
    def stimulus_df(self) -> pd.DataFrame:
        return self._stim_df
