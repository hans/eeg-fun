from importlib_resources import files
from pathlib import Path
import re
from typing import Tuple, List

import mne
import numpy as np
import pandas as pd
from pymatreader import read_mat
import scipy.io
from tqdm.auto import tqdm

from mfn400.adapters import MNEDatasetAdapter


info_re = re.compile(r"S(\d+)")

class BrennanDatasetAdapter(MNEDatasetAdapter):
    """
    Brennan et al. 2018 naturalistic N400 dataset.

    EasyCap M10 montage, 61 active electrodes, with actiCHamp amplifier.

    Data contains annotations for all words. Each annotation is of the format
    `<sentence_idx>_<word_idx>` .

    stim_df is indexed by segment_idx, sentence_idx, word_idx (all 1-indexed).
    Note that sentence_idxs do not overlap across segments.
    """

    name = "brennan2018"

    # acquisition parameters
    data_channels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
         '24', '25', '26', '27', '28', '30', '31', '32', '33', '34', '35',
         '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46',
         '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57',
         '58', '59', '60', '61']
    eog_channels = ['VEOG']
    misc_channels = ['Aux5']  # eye lead?

    # In order matching data
    all_channels = data_channels + eog_channels + misc_channels
    all_channel_types = (["eeg"] * len(data_channels)) + \
        (["eog"] * len(eog_channels)) + \
        (["misc"] * len(misc_channels))

    montage = "easycap-M10"
    sample_rate = 500
    filter_window = None

    def __init__(self, eeg_dir):
        eeg_dir = self.eeg_dir = Path(eeg_dir)

        stim_path = eeg_dir / "stimuli" / "AliceChapterOne-EEG.csv"
        self._stim_df = pd.read_csv(
            stim_path,
            index_col=None) \
            .rename(columns=dict(Position="word_idx",
                                 Sentence="sentence_idx",
                                 Segment="segment_idx")) \
            .drop(columns=["LogFreq_Prev", "LogFreq_Next"]) \
            .set_index(["segment_idx", "sentence_idx", "word_idx"])

        self._presentation_dfs = {}

        self._load_mne()

    def _load_mne(self):
        """
        Load MNE continuous representation.
        """
        self._raw_data, self._annots, self._run_ranges = {}, {}, {}
        paths = list((self.eeg_dir / "eeg").glob("S*", ))

        for subject_dir in tqdm(paths, "loading subject data"):
            subject_id = int(info_re.match(subject_dir.name).group(1).lstrip("0"))
            print(subject_id)
            self._raw_data[subject_id], self._presentation_dfs[subject_id] = \
                self._load_mne_single_subject(subject_id, subject_dir)

            # Dummy annotation: we have just one "run" per subject of continuous
            # speech stimulus.
            self._run_ranges[subject_id] = {1: (0, self._raw_data[subject_id].n_times)}

    def _load_mne_single_subject(self, subject_id: int, path) -> Tuple[mne.io.Raw, List[int]]:
        raw = mne.io.read_raw(path / ("S%02i_alice-raw.fif" % subject_id),
                              preload=True)

        # Prepare presentation df, specifying when this particular subject
        # observed each particular word.

        # Load segments; word onsets will be computed relative to segment annotation points.
        n_segment_annotations = len(raw.annotations)
        segment_data = pd.DataFrame(list(raw.annotations))
        segment_data["description"] = segment_data.description.astype(int)
        segment_data = segment_data.set_index("description")
        segment_data.index.name = "segment_idx"

        presentation_df = self.stimulus_df.copy()
        presentation_df["onset"] += segment_data.onset
        presentation_df["offset"] += segment_data.onset

        # Remove segment annotations ; we want just the words.
        raw.annotations.delete(np.arange(n_segment_annotations))

        # Add annotations based on stim_df.
        for (segment_idx, sentence_idx, word_idx), row in presentation_df.iterrows():
            raw.annotations.append(row.onset, row.offset - row.onset,
                                   description=f"{sentence_idx}_{word_idx}")

        return raw, presentation_df

    def _preprocess(self, subject_id,
                    filter_window: Tuple[float, float]) -> mne.io.Raw:
        raw = self._raw_data[subject_id]

        # Band-pass filter.
        raw = raw.filter(*filter_window)

        # Interpolate bad channels.
        raw = raw.interpolate_bads()

        return raw

    def get_presentation_data(self, subject_id) -> pd.DataFrame:
        return self._presentation_dfs[subject_id]

    @property
    def stimulus_df(self) -> pd.DataFrame:
        return self._stim_df
