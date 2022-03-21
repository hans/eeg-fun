from typings import Tuple, Optional, Dict, List

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

    def to_erp(self, epoch_window: Tuple[float, float]) -> mne.Epochs:
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

    _raw_data: Optional[Dict[int, mne.RawData]] = None
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

    def _prepare_events_seq(self) -> np.ndarray:
        """
        Prepare an MNE event-matrix representation with one event per word
        onset.
        """
        events_arr = []

        for i, run_offset in enumerate(self._run_offsets):
            word_offsets = self._stim_df.loc[i + 1].sample_id + run_offset

            events_arr.append(np.stack([
                word_offsets,
                np.zeros_like(word_offsets),
                self.word_event_id * np.ones_like(word_offsets)
            ], axis=1))

        return np.concatenate(events_arr)

    def to_erp(self, epoch_window: Tuple[float, float]) -> mne.Epochs:
        """
        Prepare the dataset for ERP analysis by epoching.
        """

        # TODO check that the data has been preprocessed before epoching.

        # Prepare an MNE event-matrix representation with one event per
        # word onset.
        events_seq = self._prepare_events_seq()

        epoch_tmin, epoch_tmax = epoch_window
        epochs = mne.Epochs(self._raw_data, events_seq, preload=True,
                            tmin=epoch_tmin, tmax=epoch_tmax)

        return epochs
