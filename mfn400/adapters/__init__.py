from pathlib import Path
from typing import Tuple, Optional, Dict, List

import mne
import numpy as np
import pandas as pd

from mfn400.n400 import prepare_erp_df


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

    def to_erp(self,
               epoch_window: Tuple[float, float],
               test_window: Tuple[float, float],
               baseline_window: Tuple[Optional[float], float] = (None, 0),
               apply_baseline=True) -> pd.DataFrame:
        """
        Prepare a dataset for ERP analysis.

        Args:
            epoch_window: Time window around each event to retain in epochs.
            test_window: Time window over which to average signal measure.
            baseline_window: Time window over which to calculate baseline.
            apply_baseline: If `True`, baseline measure will be subtracted
                from test measure in the resulting dataframe. If `False`, for
                each channel `x` in the dataframe an extra column `x_baseline`
                will also be returned, containing the baseline value.
        """
        raise NotImplementedError()


class MNEDatasetAdapter(DatasetAdapter):
    """
    Dataset adapter using internal MNE representation.

    Multiple runs are concatenated into one single continuous `RawData`
    MNE representation. `_run_ranges` stores range indices corresponding
    to meaningful item-related signal in the raw data.
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

    _run_ranges: Optional[Dict[int, Dict[int, Tuple[int, int]]]] = None
    """
    A dict of sample index ranges (into `_raw_data`) mapping item indices
    to values `[start_sample, end_sample)` (NB end is non-inclusive).
    """

    _stim_df: Optional[pd.DataFrame] = None
    """
    Columns: `token`, `token_surprisal`, `onset_time`, `offset_time`
    """

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

    def get_presentation_data(self, subject_id) -> pd.DataFrame:
        """
        Return a dataframe describing presentation to `subject_id`.

        Should minimally contain columns `item` and `onset_time`.
        """
        # By default, assume presentations are the same across subjects.
        return self.stimulus_df.copy()

    def _to_epochs_single_subject(self, subject_id,
                                  epoch_window: Tuple[float, float],
                                  baseline,
                                  **preprocessing_kwargs) -> mne.Epochs:
        raw = self._raw_data[subject_id]
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        epoch_tmin, epoch_tmax = epoch_window
        return mne.Epochs(raw, events=events, event_id=event_id,
                          tmin=epoch_tmin, tmax=epoch_tmax,
                          reject_by_annotation=False,
                          preload=True, baseline=baseline,
                          verbose=False)

    def to_epochs(self, epoch_window: Tuple[float, float],
                  baseline=(None, 0),
                  **preprocessing_kwargs) -> Dict[int, mne.Epochs]:
        """
        Prepare epoched dataset.
        """
        if not self.preprocessed:
            self.run_preprocessing(**preprocessing_kwargs)

        epochs = {
            subject_id: self._to_epochs_single_subject(
                subject_id, epoch_window, baseline=baseline,
                **preprocessing_kwargs)
            for subject_id in self._raw_data
        }
        return epochs

    def to_erp(self,
               epoch_window: Tuple[float, float],
               test_window: Tuple[float, float],
               baseline_window: Tuple[Optional[float], float] = (None, 0),
               apply_baseline=True,
               **preprocessing_kwargs) -> pd.DataFrame:
        """
        Prepare a dataset for ERP analysis.

        Args:
            epoch_window: Time window around each event to retain in epochs.
            test_window: Time window over which to average signal measure.
            baseline_window: Time window over which to calculate baseline.
            apply_baseline: If `True`, baseline measure will be subtracted
                from test measure in the resulting dataframe. If `False`, for
                each channel `x` in the dataframe an extra column `x_baseline`
                will also be returned, containing the baseline value.
        """
        # If we are not applying baseline, ask MNE to epoch without awareness
        # of baseline.
        epoch_baseline = baseline_window if apply_baseline else None
        epochs = self.to_epochs(epoch_window, epoch_baseline,
                                **preprocessing_kwargs)
        return pd.concat(
            [prepare_erp_df(epochs[subject],
                            self.get_presentation_data(subject),
                            test_window=test_window,
                            apply_baseline=apply_baseline,
                            baseline=baseline_window) \
             .drop(columns=["subject_idx"], errors="ignore")
             for subject in epochs],
            names=["subject_idx"],
            keys=[int(idx) for idx in epochs.keys()]
        )

    def to_cdr(self, x_path, y_path, **preprocessing_kwargs):
        """
        Convert this dataset to a CDR-friendly representation. Save at the
        given paths.
        """

        if not self.preprocessed:
            self.run_preprocessing(**preprocessing_kwargs)

        # Write X data.
        X_df = pd.concat({subject_idx: self.get_presentation_data(subject_idx)
                          for subject_idx in self._raw_data.keys()},
                         names=["subject", "index"])
        # Set expected CDR `time` column.
        # TODO adjust time values correctly -- for each item, subtract max
        # time (onset+duration) of preceding item. Probably should go in
        # get_presentation_data
        X_df["time"] = X_df["onset_time"]

        # Drop NA values.
        # TODO check / log effects
        X_df = X_df.dropna()

        X_df.to_csv(x_path, sep=" ")

        # Write Y data.
        for subject_id, raw_data in self._raw_data.items():
            df = raw_data.to_data_frame(time_format=None)
            run_ranges = self._run_ranges[subject_id]

            # Split signal segments by item.
            run_dfs = {item_idx: df.loc[start_idx:end_idx]
                       for item_idx, (start_idx, end_idx)
                       in run_ranges.items()}

            df = pd.concat(run_dfs, names=["item", "run_sample_id"]) \
                .dropna()
            df["subject"] = subject_id

            # Write header once.
            header = None if Path(y_path).exists() else True

            df.to_csv(y_path, mode="a", sep=" ", header=header,
                      float_format="%.4f")
