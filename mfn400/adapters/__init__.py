from pathlib import Path
from typing import Tuple, Optional, Dict, List

import mne
import numpy as np
import pandas as pd


class DatasetAdapter(object):
    """
    Handles conversions between arbitrary N400 language dataset and
    analysis-specific data formats (for ERP, CDR, etc.).
    """

    name = "abstract"

    @property
    def stimulus_df(self) -> pd.DataFrame:
        raise NotImplementedError()

    def to_cdr(self, x_path, y_path):
        """
        Convert this dataset to a CDR-friendly representation. Save at the
        given paths.
        """
        raise NotImplementedError()

    def to_erp(self, epoch_window: Tuple[float, float]) -> Dict[int, mne.Epochs]:
        """
        Prepare the dataset for ERP analysis by epoching.
        """
        raise NotImplementedError()


class MNEDatasetAdapter(DatasetAdapter):
    """
    Dataset adapter using internal MNE representation.

    Multiple runs are concatenated into one single continuous `RawData`
    MNE representation. `_run_offsets` stores the index of the first sample
    of each run in this raw data.
    """

    word_event_id = 2
    preprocessed = False

    _raw_data: Optional[Dict[int, mne.io.Raw]] = None
    """
    A continuous-time representation of the EEG data, i.e. artificially
    merging separate runs into a continuous stream. Per MNE norms there
    is a "bad" annotation at the boundaries of different runs, to prevent
    merging data across runs when filtering, epoching, etc.

    Mapping from subject -> RawData.
    """

    _run_offsets: Optional[Dict[int, List[int]]] = None
    """
    A list of sample indices (into `_raw_data`) describing the first sample
    of each run.
    """

    _stim_df: Optional[pd.DataFrame] = None
    """
    Columns: `token`, `token_surprisal`, `onset_time`, `offset_time`
    """

    # TODO bring over other methods from broderick2018

    def _prepare_events_seq(self, subject_idx) -> np.ndarray:
        """
        Prepare an MNE event-matrix representation with one event per word
        onset.
        """
        events_arr = []
        run_offsets = self._run_offsets[subject_idx]

        for i, run_offset in enumerate(run_offsets):
            word_offsets = self._stim_df.loc[i + 1].sample_id + run_offset

            events_arr.append(np.stack([
                word_offsets,
                np.zeros_like(word_offsets),
                self.word_event_id * np.ones_like(word_offsets)
            ], axis=1))

        return np.concatenate(events_arr)

    def _preprocess(self, raw_data: mne.io.Raw, **kwargs) -> mne.io.Raw:
        raise NotImplementedError()

    def run_preprocessing(self, **kwargs):
        """
        Run any necessary preprocessing procedure on EEG data.
        Subclasses should override `_preprocess`.
        """
        if self.preprocessed:
            raise RuntimeError("run_preprocessing was already called.")

        for idx in self._raw_data:
            self._raw_data[idx] = self._preprocess(self._raw_data[idx],
                                                   **kwargs)

        self.preprocessed = True

    def to_erp(self, epoch_window: Tuple[float, float],
            **preprocessing_kwargs) -> Dict[int, mne.Epochs]:
        """
        Prepare the dataset for ERP analysis by epoching.
        """

        if not self.preprocessed:
            self.run_preprocessing(**preprocessing_kwargs)

        epochs = {}
        for subject_idx, raw_data in self._raw_data.items():
            # Prepare an MNE event-matrix representation with one event per
            # word onset.
            events_seq = self._prepare_events_seq(subject_idx)

            epoch_tmin, epoch_tmax = epoch_window
            epochs[subject_idx] = \
                mne.Epochs(raw_data, events_seq, preload=True,
                           tmin=epoch_tmin, tmax=epoch_tmax)

        return epochs

    def to_cdr(self, x_path, y_path, **preprocessing_kwargs):
        """
        Convert this dataset to a CDR-friendly representation. Save at the
        given paths.
        """

        if not self.preprocessed:
            self.run_preprocessing(**preprocessing_kwargs)

        # Write X data.
        # Need to replicate for each subject to be compatible with CDR.
        X_df = pd.concat({subject_idx: self.stimulus_df.reset_index()
                          for subject_idx in self._raw_data.keys()},
                         names=["subject"])
        # Set expected CDR `time` column
        X_df["time"] = X_df["onset_time"]
        X_df.to_csv(x_path, sep=" ")

        # Write Y data.
        for subject_id, raw_data in self._raw_data.items():
            df = raw_data.to_data_frame(time_format=None)
            run_offsets = self._run_offsets[subject_id]

            # Undo concatenation into a single raw array, so that each
            # participant-run begins at time t=0.
            run_dfs = [df.loc[start_idx:end_idx] for start_idx, end_idx
                       in zip(run_offsets, run_offsets[1:] + [len(df)])]
            run_dfs = [run_df.assign(time=run_df.time - run_df.time.min())
                       for run_df in run_dfs]

            df = pd.concat(run_dfs, keys=[i + 1 for i in range(len(run_dfs))],
                           names=["run"], ignore_index=True)
            df["subject"] = subject_id

            # Write header once.
            header = None if Path(y_path).exists() else True

            df.to_csv(y_path, mode="a", sep=" ", header=header,
                      float_format="%.4f")
