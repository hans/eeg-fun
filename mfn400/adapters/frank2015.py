import itertools
from pathlib import Path
import re
from typing import Optional, Tuple, List, Dict

import mne
import numpy as np
import pandas as pd
import scipy.io

from mfn400.adapters import MNEDatasetAdapter, DatasetAdapter
from mfn400.mne_utils import annotate_given_breaks


info_re = re.compile(r"EEG(\d+)\.set")


class FrankDatasetAdapter(MNEDatasetAdapter):
    """
    Frank et al. 2015 naturalistic N400 dataset.
    """

    name = "frank2015"

    # acquisition parameters
    data_channels = ... # [f"V{x}" for x in range(1, 129)]
    # two mastoid references
    reference_channels = ...  # ["M1", "M2"]
    sample_rate = 250

    def __init__(self, eeg_dir, stim_path):
        eeg_dir = Path(eeg_dir)
        self._prepare_paths(eeg_dir)

        self._stim_df = pd.read_csv(stim_path)

        self._load_mne()

    def _prepare_paths(self, eeg_dir):
        eeg_paths = sorted(eeg_dir.glob("*.set"))
        eeg_paths = {info_re.match(p.name).group(1).lstrip(0): p
                     for p in eeg_paths}
        self._eeg_paths = eeg_paths

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

    def _load_mne_single_subject(self, subject_id, run_path) -> Tuple[mne.io.Raw, List[int]]:
        raw = mne.io.read_raw_eeglab(run_path, preload=True)

        # Break periods between items are represented in the raw data as NaN
        # data. Detect these and add MNE "BAD" annotations.
        raw_data: np.ndarray = raw.get_data()
        presentation_begins = np.where(~np.isnan(raw_data[0, 1:])
                                       & np.isnan(raw_data[0, :-1]))[0]
        presentation_ends = np.where(~np.isnan(raw_data[0, :-1])
                                     & np.isnan(raw_data[0, 1:]))[0]
        presentation_spans = np.array(list(zip(presentation_begins,
                                               presentation_ends)))

        annotate_given_breaks(raw, presentation_spans)

        return raw_data, presentation_begins

    def get_presentation_data(self, subject_id) -> pd.DataFrame:
        raw = self._raw_data[subject_id]
        annotations_df = pd.DataFrame({
            "description": raw.annotations.description.astype(int),
            "onset": raw.annotations.onset
        })

        # Compute sentence/word index from funky representation in annotation
        # data.
        #
        # From their readme:
        #
        # If the value is larger than 50, then the word was the first word of
        # the sentence with ID number EEG.event(n).type-50 (e.g., type 51 is the first word of sentence 1).
        # If the value is between 2 and 15, then it identifies the word position within the sentence (e.g., type 2
        # is the second word of the current sentence).

        annotations_df.loc[annotations_df.description <= 50, "word_idx"] = \
            annotations_df.description - 1
        annotations_df.loc[annotations_df.description > 50, "word_idx"] = 0
        annotations_df.loc[annotations_df.description > 50, "sentence_idx"] = \
            annotations_df.description - 50
        annotations_df["sentence_idx"] = annotations_df.sentence_idx.fillna(method="ffill")
        annotations_df = annotations_df.astype({"word_idx": int, "sentence_idx": int}) \
            .drop(columns=["description"])

        ret = pd.merge(annotations_df, self.stimulus_df,
                       left_on=["sentence_idx", "word_idx"], right_index=True) \
            .rename(dict(onset="time"))

        return ret

    def _preprocess(self, raw_data: mne.io.Raw,
                    filter_window: Tuple[float, float]) -> mne.io.Raw:
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
