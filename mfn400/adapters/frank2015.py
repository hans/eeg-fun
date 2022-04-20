import contextlib
from functools import reduce
import itertools
import logging
from pathlib import Path
import re
from typing import Optional, Tuple, List, Dict

import mne
import numpy as np
import pandas as pd
import pymatreader
import scipy.io

from mfn400.adapters import MNEDatasetAdapter, DatasetAdapter
from mfn400.mne_utils import annotate_given_breaks


info_re = re.compile(r"EEG(\d+)\.set")

SUBJECT_6_MISSING_SENTENCES = [ 27, 173, 171, 145, 147,  39,
                               185,  46,  76,  68,  61 ]
"""
Sentences (1-based index) missing from EEG recording of subject 6, even
after merging the two parts stored separately. These IDs are still present
in the e.g. stimulus data of subject 6, which is problematic.
"""

DROP_REGIONS = {
    "6": [150],
    "8": [31],
    "10": [77],
    "14": [0],
    "17": [52],
    "20": [0],
}
"""
Some subjects have regions of signal data without corresponding sentence
annotations. This metadata is stored here (designating 0-based region index in 
raw signal) and applied in `_load_mne_single_subject`.

See information in test suite `test_item_onset_agreement` for why this was
done.
"""

# Frank data contains pre-computed response data for a variety of ERPs. We're
# just extracting N400 data here.
DATA_COMPONENTS = ["ELAN", "LAN", "N400", "EPNP", "P600", "PNP"]
N400_COMPONENT_IDX = DATA_COMPONENTS.index("N400")


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

        self._stim_df = pd.read_csv(
            stim_path,
            index_col=["subject_idx", "sentence_idx", "word_idx"])

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
        self._raw_data, self._run_ranges = {}, {}
        for subject_id, run_paths in self._eeg_paths.items():
            self._raw_data[subject_id], presentation_spans = \
                self._load_mne_single_subject(subject_id, run_paths)
            self._run_ranges[subject_id] = \
                self._prepare_run_ranges(subject_id, presentation_spans)

    def _load_mne_single_subject(self, subject_id, run_path) -> Tuple[mne.io.Raw, List[int]]:
        kwargs = dict(eog=self.eog_channels, preload=True)
        if subject_id == "6":
            # TODO document
            raw = read_two_part(run_path, **kwargs)
        else:
            raw = mne.io.read_raw_eeglab(run_path, **kwargs)
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

        # BUG want to annotate the complement of the span set denoted by
        # presentation_spans
        annotate_given_breaks(raw, presentation_spans)
        
        # Some subjects have data regions missing annotations. This causes
        # alignment issues downstream. So we'll drop these regions from
        # `presentation_spans` before it becomes an issue.
        if subject_id in DROP_REGIONS:
            presentation_spans = np.delete(
                presentation_spans, DROP_REGIONS[subject_id], axis=0)

        return raw, presentation_spans

    def _prepare_run_ranges(self, subject_id, presentation_spans):
        """
        Determine sample range corresponding to each run/item, working back
        from annotated raw data and the data regions `presentation_spans`
        as returned by `_load_mne_single_subject`.
        """
        
        # `presentation_spans` above is following the order of presentation to
        # subject. Map this back to item idx.
        span_df = pd.DataFrame(
            presentation_spans,
            columns=["start_sample", "end_sample"])

        sentence_idxs = self._get_sentence_presentation_order(subject_id)
        
        # DEV: Code for checking match between item sample boundaries as 
        # determined by directly analyzing signal (in `span_df`) and item
        # sample boundaries as determined by existing annotations.
        #
        # df1 = (span_df.start_sample / self.sample_rate)
        # annotations_df = self._get_annotations_df(subject_id)
        # annotations_df = annotations_df[annotations_df.sentence_idx != -1]
        # df2 = annotations_df.groupby("sentence_idx", sort=False).onset.agg(onset="min").reset_index()
        # dfx = pd.concat([df1, df2], axis=1)
        # dfx["onset_diff"] = dfx.onset - dfx.start_sample
        # print(dfx)
        # 
        # ^ verify that onset_diff is always .104 for all items. looks good.

        assert len(sentence_idxs) == len(span_df), subject_id

        span_df["sentence_idx"] = sentence_idxs
        return {
            row.sentence_idx: (row.start_sample, row.end_sample)
            for _, row in span_df.iterrows()
        }

    def _get_annotations_df(self, subject_id) -> pd.DataFrame:
        """
        Build a dataframe with readable sentence- and word-annotations
        describing the presentation to a particular subject.
        """
        raw = self._raw_data[subject_id]
        annotations_df = pd.DataFrame({
            "description": raw.annotations.description,
            "onset": raw.annotations.onset
        })

        # Retain just presentation data (not BAD annotations).
        annotations_df = annotations_df.loc[~annotations_df.description.str.startswith(("BAD", "EDGE"))] \
            .astype({"description": int})

        # Compute sentence/word index from funky representation in annotation
        # data.
        #
        # From their readme:
        #
        # If the value is larger than 50, then the word was the first word of
        # the sentence with ID number EEG.event(n).type-50 (e.g., type 51 is
        # the first word of sentence 1).
        # If the value is between 2 and 15, then it identifies the word
        # position within the sentence (e.g., type 2 is the second word of the
        # current sentence).

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
        annotations_df["subject_idx"] = int(subject_id)

        return annotations_df

    def _get_sentence_presentation_order(self, subject_id) -> List[int]:
        """
        Return a list of sentence IDs (1-indexed) in the order presented to the
        given subject.
        """
        annotations_df = self._get_annotations_df(subject_id)
        annotations_df = annotations_df[annotations_df.sentence_idx != -1]
        return list(annotations_df.sentence_idx.unique())

    def get_presentation_data(self, subject_id) -> pd.DataFrame:
        annotations_df = self._get_annotations_df(subject_id)

        ret = pd.merge(annotations_df, self.stimulus_df, how="left",
                       left_on=["subject_idx", "sentence_idx", "word_idx"],
                       right_index=True) \
            .rename(columns=dict(onset="onset_time"))

        ret["item"] = ret["sentence_idx"]

        return ret

    def _preprocess(self, raw_data: mne.io.Raw) -> mne.io.Raw:
        # Signals are already filtered and rereferenced in published data.
        return raw_data

    @property
    def stimulus_df(self) -> pd.DataFrame:
        return self._stim_df


# class RawEEGLABFromStruct(mne.io.eeglab.RawEEGLab):
#     """
#     Hacky subclass to construct from preloaded mat.
#     """

#     @verbose
#     def __init__(self, eeg, eog=(),
#                  preload=False, uint16_codec=None, verbose=None):  # noqa: D102
#         if eeg.trials != 1:
#             raise TypeError('The number of trials is %d. It must be 1 for raw'
#                             ' files. Please use `mne.io.read_epochs_eeglab` if'
#                             ' the .set file contains epochs.' % eeg.trials)

#         last_samps = [eeg.pnts - 1]
#         info, eeg_montage, _ = _get_info(eeg, eog=eog)

#         # read the data
#         assert not isinstance(eeg.data, str)

#         if preload is False or isinstance(preload, str):
#             warn('Data will be preloaded. preload=False or a string '
#                  'preload is not supported when the data is stored in '
#                  'the .set file')
#         # can't be done in standard way with preload=True because of
#         # different reading path (.set file)
#         if eeg.nbchan == 1 and len(eeg.data.shape) == 1:
#             n_chan, n_times = [1, eeg.data.shape[0]]
#         else:
#             n_chan, n_times = eeg.data.shape
#         data = np.empty((n_chan, n_times), dtype=float)
#         data[:n_chan] = eeg.data
#         data *= CAL
#         # Jump to super-super-class
#         # NB removed filenames kwarg
#         super(RawEEGLAB, self).__init__(
#             info, data, filenames=["dummy"], last_samps=last_samps,
#             orig_format='double', verbose=verbose)

#         # create event_ch from annotations
#         annot = read_annotations(input_fname)
#         self.set_annotations(annot)
#         _check_boundary(annot, None)

#         _set_dig_montage_in_init(self, eeg_montage)

#         latencies = np.round(annot.onset * self.info['sfreq'])
#         _check_latencies(latencies)

#     def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
#         """Read a chunk of raw data."""
#         _read_segments_file(
#             self, data, idx, fi, start, stop, cals, mult, dtype='<f4')


@contextlib.contextmanager
def monkeypatched(object, name, patch):
    """ Temporarily monkeypatches an object. """
    pre_patched_value = getattr(object, name)
    setattr(object, name, patch)
    yield object
    setattr(object, name, pre_patched_value)


def make_eeglab_loader(struct_key):
    def load(fname, uint16_codec):
        """Check if the mat struct contains 'EEG'."""
        # Stolen from mne.io.eeglab.eeglab, but using a variable `struct_key`
        # rather than fixed "EEG"

        from mne.utils import Bunch

        eeg = pymatreader.read_mat(fname, uint16_codec=uint16_codec)
        if 'ALLEEG' in eeg:
            raise NotImplementedError(
                'Loading an ALLEEG array is not supported. Please contact'
                'mne-python developers for more information.')
        if struct_key in eeg:  # fields are contained in EEG structure
            eeg = eeg[struct_key]
        eeg = eeg.get('EEG', eeg)  # handle nested EEG structure
        eeg = Bunch(**eeg)
        eeg.trials = int(eeg.trials)
        eeg.nbchan = int(eeg.nbchan)
        eeg.pnts = int(eeg.pnts)
        return eeg

    return load


def read_two_part(path, **kwargs):
    """
    Read two-part EEGLAB output for participant 6.

    Stored in keys `EEG` and `EEG_part2`.
    """

    part1 = mne.io.read_raw_eeglab(path, **kwargs)

    monkey_loader = make_eeglab_loader("EEG_part2")
    with monkeypatched(mne.io.eeglab.eeglab, "_check_load_mat", monkey_loader):
        part2 = mne.io.read_raw_eeglab(path, **kwargs)

    return mne.concatenate_raws([part1, part2])


### Utilities for working with `stimuli_erp.mat`
def make_signal_df(data, key, target_key=None):
    """
    Build dataframe describing ERP signal from Frank formatted struct.
    """
    target_key = target_key or key
    return pd.concat(
        {sentence_idx: pd.DataFrame(mat[:, :, N400_COMPONENT_IDX],
                                    index=pd.RangeIndex(mat.shape[0], name="word_idx")) \
            .reset_index().melt(id_vars=["word_idx"], var_name="subject_idx",
                                value_name=target_key)
            for sentence_idx, mat in enumerate(data[key])},
        names=["sentence_idx", "idx"]) \
        .reset_index().drop(columns=["idx"]) \
        .set_index(["subject_idx", "sentence_idx", "word_idx"])


def make_feature_df(data, key, target_key=None, final_axis_name="subject_idx"):
    target_key = target_key or key
    return pd.concat(
        {sentence_idx: pd.DataFrame(mat,
                                    index=pd.RangeIndex(mat.shape[0], name="word_idx")) \
                                    .reset_index().melt(
                                        id_vars=["word_idx"],
                                        var_name=final_axis_name,
                                        value_name=target_key)
         for sentence_idx, mat in enumerate(data[key])},
        names=["sentence_idx", "idx"]) \
        .reset_index().drop(columns=["idx"]) \
        .set_index(["subject_idx", "sentence_idx", "word_idx"])


def make_surprisal_df(data, key, target_key=None, i_offset=0):
    target_key = target_key or key
    return pd.concat(
        {sentence_idx: pd.DataFrame(mat, index=pd.RangeIndex(mat.shape[0], name="word_idx"),
                                    columns=[f"{target_key}_{i_offset+idx}" for idx in range(mat.shape[1])])
         for sentence_idx, mat in enumerate(data[key])},
        names=["sentence_idx", "word_idx"])


def make_control_df(data, key):
    return pd.concat([pd.DataFrame({key: wf}, index=pd.RangeIndex(len(wf), name="word_idx"))
                      for idx, wf in enumerate(data[key])],
                     names=["sentence_idx"], keys=np.arange(len(data[key])))