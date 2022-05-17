from pathlib import Path
import re
from typing import Tuple, List

import mne
import numpy as np
import pandas as pd
from pymatreader import read_mat
import scipy.io

from mfn400.adapters import MNEDatasetAdapter


info_re = re.compile(r"S(\d+)\.mat")


# Prepare MNE info representation
# Copied from Eelbrain https://github.com/Eelbrain/Alice/blob/main/import_dataset/convert-all.py
def build_mne_info():
    ch_default = {
      'scanno': 307,
      'logno': 1,
      'kind': 3,
      'range': 1.0,
      'cal': 1.0,
      'coil_type': 0,
      'loc': numpy.array([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]),
      'unit': 107,
      'unit_mul': 0,
      'coord_frame': 0,
    }

    samplingrate = 500
    montage = mne.channels.read_custom_montage('easycapM10-acti61_elec.sfp')
    # montage.plot()
    info = mne.create_info(montage.ch_names, samplingrate, 'eeg')
    info.set_montage(montage)
    info['highpass'] = 0.1
    info['lowpass'] = 200
    for ch_name in ['VEOG', 'Aux5', 'AUD']:
        info['chs'].append({**ch_default, 'ch_name': ch_name})
        info['ch_names'].append(ch_name)
        info['nchan'] += 1

    return info


def get_usable_subjects()



class BrennanDatasetAdapter(MNEDatasetAdapter):
    """
    Brennan et al. 2018 naturalistic N400 dataset.

    EasyCap M10 Acti61 montage.

    Recordings by subject are split into 'segments.'
    Time annotations are reset to 0 at the start of each segment, but not
    between sentences / words.

    stim_df is indexed by segment_idx, sentence_idx, word_idx
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
        eeg_dir = Path(eeg_dir)

        self.use_subjects = self._get_usable_subjects(eeg_dir)
        """subjects which should be used, according to dataset annotations"""

        self._prepare_paths(eeg_dir)

        stim_path = eeg_dir / "AliceChapterOne-EEG.csv"
        self._stim_df = pd.read_csv(
            stim_path,
            index_col=None) \
            .rename(columns=dict(Position="word_idx",
                                 Sentence="sentence_idx",
                                 Segment="segment_idx")) \
            # Ignore redundant lagged columns
            .drop(columns=["LogFreq_Prev", "LogFreq_Next"]) \
            .set_index(["segment_idx", "sentence_idx", "word_idx"])
        # Onsets are reset between segments. Fix this.
        # TODO make sure we're doing this correctly -- using offset of previous segment.
        segment_onsets = self._stim_df.groupby("segment_idx").offset.max() \
            .shift(1).fillna(0.).cumsum()
        self._stim_df.onset += segment_onsets
        self._stim_df.offset += segment_onsets
        # Segments are no longer useful. Drop.
        self._stim_df = self._stim_df.droplevel("segment_idx")

        self._load_mne()

    def _get_usable_subjects(self, eeg_dir) -> List[int]:
        datasets = read_mat(eeg_dir / "datasets.mat")
        return [int(s[1:3].lstrip("0")) for s in datasets["use"]]

    def _prepare_paths(self, eeg_dir):
        eeg_paths = sorted(eeg_dir.glob("S*.mat"))
        eeg_paths = {info_re.match(p.name).group(1).lstrip("0"): p
                     for p in eeg_paths}

        eeg_paths = {idx: path for idx, path in eeg_paths.items()
                     if idx in self.use_subjects}

        self._eeg_paths = eeg_paths

    def _load_mne(self):
        """
        Load MNE continuous representation.
        """
        self._raw_data, self._annots, self._run_ranges = {}, {}, {}
        for subject_id, run_paths in self._eeg_paths.items():
            self._raw_data[subject_id], self._annots[subject_id] = \
                self._load_mne_single_subject(subject_id, run_paths)

            # Dummy annotation: we have just one "run" per subject of continuous
            # speech stimulus.
            self._run_ranges[subject_id] = {1: (0, self._raw_data[subject_id].n_times)}

    def _load_mne_single_subject(self, subject_id, run_path) -> Tuple[mne.io.Raw, List[int]]:
        info = mne.create_info(ch_names=self.all_channels,
                               sfreq=self.sample_rate,
                               ch_types=self.all_channel_types)
        raw = mne.io.read_raw_fieldtrip(run_path, info=info, data_name="raw")

        # Add annotations based on stim_df.
        for (sentence_idx, word_idx), row in self.stimulus_df.iterrows():
            raw.annotations.append(row.onset, row.offset - row.onset,
                                   description=f"{sentence_idx}_{word_idx}")

        # Load annotations from authors' preprocessing.
        annots_path = run_path.parent / "proc" / run_path.name
        assert annots_path.exists(), f"Missing proc/{run_path.name}"
        annots = scipy.io.loadmat(annots_path, simplify_cells=True)["proc"]

        return raw, annots

    def _preprocess(self, subject_id) -> mne.io.Raw:
        raw = self._raw_data[subject_id]
        annots = self._annots[subject_id]

        # TODO filtering?

        # Re-reference
        reference_channels = set(annots["refchannels"]) & set(raw.info["ch_names"])
        if len(reference_channels) > 0:
            mne.set_eeg_reference(raw, reference_channels)

        # Subset channels, based on their manual analysis
        pick_channels = annots["rejections"]["final"]["chanpicks"]
        raw = raw.pick_channels(pick_channels)

        return raw

    def get_presentation_data(self, subject_id) -> pd.DataFrame:
        return self.stimulus_df.copy()

    @property
    def stimulus_df(self) -> pd.DataFrame:
        return self._stim_df
