import itertools
from pathlib import Path
import re
from typings import Optional, Tuple, List, Dict

import mne
import numpy as np
import pandas as pd
import scipy.io

from mfn400.adapters import MNEDatasetAdapter, DatasetAdapter


info_re = re.compile(r"Subject(\d+)_Run(\d+)\.mat")


class BroderickDatasetAdapter(MNEDatasetAdapter):
    """
    Broderick et al. 2018 naturalistic N400 dataset.
    """

    name = "broderick2018"

    # acquisition parameters
    data_channels = ...
    reference_channels = ...
    sample_rate = 128

    def __init__(self, eeg_dir, stim_dir):
        self._prepare_stim_paths(eeg_dir, stim_dir)

        self._raw_data: Optional[Dict[int, mne.RawData]] = None
        """
        A continuous-time representation of the EEG data, i.e. artificially
        merging separate runs into a continuous stream. Per MNE norms there
        is a "bad" annotation at the boundaries of different runs, to prevent
        merging data across runs when filtering, epoching, etc.

        Mapping from subject -> RawData.
        """

        self._run_offsets: Optional[Dict[int, List[int]]] = None
        """
        A list of sample indices (into `_raw_data`) describing the first sample
        of each run.
        """

        self._stim_df: Optional[pd.DataFrame] = None
        """
        Columns: `token`, `token_surprisal`, `onset_time`, `offset_time`
        """

    def _prepare_stim_paths(self, eeg_dir, stim_dir):
        # TODO
        ...
        self._eeg_paths: Dict[int, Dict[int, Path]] = {}
        self._stim_paths: Dict[int, Path] = {}

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

    def _preprocess_mne(self, filter_window: Tuple[float, float]):
        # TODO integrate

        # Set reference.
        # TODO(EEG) is this reference right? just using the average of bilateral channels
        self._raw_data = self._raw_data.set_eeg_reference(self.reference_channels)

        # Run band-pass filter.
        self._raw_data = self._raw_data.filter(*filter_window)

    @property
    def stimulus_df(self) -> pd.DataFrame:
        raise NotImplementedError()

    def to_cdr(self, x_path, y_path):
        """
        Convert this dataset to a CDR-friendly representation. Save at the
        given paths.
        """
        for subject_id, raw_data in self._raw_data.items():
            # TODO write x data.

            df = raw_data.to_data_frame(time_format=None)
            run_offsets = self._run_offsets[subject_id]

            # Undo concatenation into a single raw array, so that each
            # participant-run begins at time t=0.
            run_dfs = [df.loc[start_idx:end_idx] for start_idx, end_idx
                       in zip(run_offsets, run_offsets[1:] + [len(df)])]
            run_dfs = [run_df.assign(time=run_df.time - run_df.time.min())
                       for run_df in run_dfs]

            df = pd.concat(run_dfs, keys=[i + 1 for i in range(len(run_dfs))],
                           names=["run"])
            df["subject"] = subject_id

            # Write header once.
            header = None if Path(y_path).exists() else True

            df.to_csv(y_path, mode="a", sep=" ", header=header)
