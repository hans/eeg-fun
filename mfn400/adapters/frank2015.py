import itertools
import logging
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
    
    Data as published is already preprocessed:
    - 0.05-25Hz band-pass filter
    - re-referenced to average of left and right mastoid electrodes
    
    Channel names follow EasyCap M10 montage.
    """

    name = "frank2015"

    # acquisition parameters
    data_channels = ['1', '10', '12', '14', '16', '18', '21', '22', '24', 
                     '25', '26', '29', '30', '31', '33', '34', '35', '36', 
                     '37', '38', '39', '40', '41', '42', '44', '45', '46', 
                     '47', '48', '49', '50', '8']
    eog_channels = ["VEOG", "HEOG"]
    montage = "easycap-M10"
    sample_rate = 250
    filter_window = (0.05, 25.)

    def __init__(self, eeg_dir, stim_path):
        eeg_dir = Path(eeg_dir)
        self._prepare_paths(eeg_dir)

        self._stim_df = pd.read_csv(stim_path, index_col=0) \
            .set_index(["sentence_idx", "word_idx"])

        self._load_mne()

    def _prepare_paths(self, eeg_dir):
        eeg_paths = sorted(eeg_dir.glob("*.set"))
        eeg_paths = {info_re.match(p.name).group(1).lstrip("0"): p
                     for p in eeg_paths}
        
        self._eeg_paths = eeg_paths

    def _load_mne(self):
        """
        Load MNE continuous representation.
        """
        raw_data, run_ranges = {}, {}
        for subject_id, run_paths in self._eeg_paths.items():
            raw_data[subject_id], run_ranges[subject_id] = \
                self._load_mne_single_subject(subject_id, run_paths)

        self._raw_data = raw_data
        self._run_ranges = run_ranges

    def _load_mne_single_subject(self, subject_id, run_path) -> Tuple[mne.io.Raw, List[int]]:
        raw = mne.io.read_raw_eeglab(run_path, preload=True,
                                     eog=self.eog_channels)
        raw.set_montage(self.montage)
        
        # Not sure if it's important to have the existing band-pass filter
        # information in the MNE info. Probably not.
        # raw.info["highpass"], raw.info["lowpass"] = self.filter_window
        
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

        return raw, presentation_spans

    def get_presentation_data(self, subject_id) -> pd.DataFrame:
        raw = self._raw_data[subject_id]
        annotations_df = pd.DataFrame({
            "description": raw.annotations.description,
            "onset": raw.annotations.onset
        })
        
        # Retain just presentation data (not BAD annotations).
        annotations_df = annotations_df.loc[~annotations_df.description.str.startswith("BAD")] \
            .astype({"description": int})

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

        # Some subjects have leading words before a sentence marker. Don't
        # know what happened in the data here .. but let's just mark these
        # rows as bad.
        na_sentences = annotations_df.sentence_idx.isna().sum()
        if na_sentences > 0:
            logging.warn(f"Subject {subject_id} had {na_sentences} "
                          "word annotations with no sentence identifier. "
                          "Setting sentence_idx = -1, make sure to ignore.")
            annotations_df.loc[annotations_df.sentence_idx.isna(), "sentence_idx"] = -1

        annotations_df = annotations_df.astype({"word_idx": int, "sentence_idx": int}) \
            .drop(columns=["description"])

        ret = pd.merge(annotations_df, self.stimulus_df, how="left",
                       left_on=["sentence_idx", "word_idx"], right_index=True) \
            .rename(columns=dict(onset="onset_time"))
        
        ret["item"] = ret["sentence_idx"]

        return ret

    def _preprocess(self, raw_data: mne.io.Raw) -> mne.io.Raw:
        # Signals are already filtered and rereferenced in published data.
        return raw_data

    @property
    def stimulus_df(self) -> pd.DataFrame:
        return self._stim_df
