"""
Extra utilities for dealing with MNE data structures.
"""

from typing import List, Tuple

import mne
import numpy as np


# def crop_raw(raw: mne.io.Raw, regions: List[Tuple[int, int]],
#              check_non_overlapping=True) -> mne.io.Raw:
#     """
#     Select subregions of the given Raw instance.
#     Returns a new `Raw` instance containing the concatenated subregions, with
#     "bad" annotations at the boundaries.
#
#     `regions` denotes start and end sample indices (inclusive on both ends).
#     """
#
#     regions = np.array(regions)
#     if check_non_overlapping and (regions[:-1, 1] > regions[1:, 0]).any():
#         raise ValueError("Provided regions are overlapping.")
#     if (regions < 0).any() or (regions > raw.n_times).any():
#         raise ValueError("Provided regions are out of bounds "
#                          f"[0, {raw.n_times}]")
#
#     data: np.ndarray = raw.get_data()
#     ret = mne.concatenate_raws([
#         mne.io.RawArray(data[:, begin:end + 1], raw.info, verbose=False)
#         for begin, end in regions
#     ])
#
#     # TODO transfer annotations/events
#
#     # DEV: gave up on this and went for annotate_break instead


def combine_annotations(x: mne.Annotations, y: mne.Annotations) -> mne.Annotations:
    """
    Combine two Annotations sequences. Unlike the MNE implementation of
    `_combine_annotations`, this function doesn't modify any times / durations.
    """
    # TODO do these need to be sorted?
    onset = np.concatenate([x.onset, y.onset])
    duration = np.concatenate([x.duration, y.duration])
    description = np.concatenate([x.description, y.description])
    ch_names = np.concatenate([x.ch_names, y.ch_names])
    return mne.Annotations(onset, duration, description, x.orig_time, ch_names)


def annotate_given_breaks(raw: mne.io.Raw, regions: List[Tuple[int, int]],
                          check_non_overlapping=True) -> mne.io.Raw:
    """
    Annotate specific regions of samples as breaks inplace.

    `regions` denotes start and end sample indices (inclusive on both ends).
    """
    regions = np.array(regions)
    if check_non_overlapping and (regions[:-1, 1] > regions[1:, 0]).any():
        raise ValueError("Provided regions are overlapping.")
    if (regions < 0).any() or (regions > raw.n_times).any():
        raise ValueError("Provided regions are out of bounds "
                         f"[0, {raw.n_times}]")

    # Determine annotation time ranges, converting from sample indices
    sample_rate = raw.info["sfreq"]
    annotations = mne.Annotations(
        onset=regions[:, 0] / sample_rate,
        duration=(regions[:, 1] - regions[:, 0]) / sample_rate,
        description=["BAD_break"],
        orig_time=raw.info["meas_date"],
    )

    raw.set_annotations(combine_annotations(raw.annotations, annotations))

    return raw
