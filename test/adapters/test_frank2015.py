from pathlib import Path

import pytest

from mfn400.adapters.frank2015 import FrankDatasetAdapter


@pytest.fixture(scope="session")
def frank_dataset():
    ret = Path("/om/data/public/language-eeg/frank2015")
    assert ret.exists()
    return ret


@pytest.fixture(scope="session")
def frank_processed_stimuli():
    ret = Path("./output/frank2015/stim_df.csv")
    assert ret.exists()
    return ret


@pytest.fixture
def frank2015(frank_dataset, frank_processed_stimuli):
    return FrankDatasetAdapter(frank_dataset, frank_processed_stimuli)


def test_item_onset_agreement(frank2015):
    """
    There are two ways of determining the onset of an item in the signal
    stream:

        1. Look for boundaries between regions of NaN signal and non-NaN
           signal. This is the principle method in
           FrankDatasetAdapter._load_mne_single_subject.
        2. Use the authors' annotations stored in the EEGLAB data, which
           specify both word and item-level onsets.

    These two methods don't always agree. There are some signal regions that
    are missing annotations. We need to make sure these are dropped in
    preprocessing so that things don't break when we try to align annotation
    data with raw data later downstream.

    It's easy to check that the data is well cleaned: the two item onset
    results should return 1) the same number of items and 2) similar onset
    guesses.
    """

    for subject_id, run_paths in frank2015._eeg_paths.items():
        # Re-run _load_mne_single_subject to get access to raw
        # presentation_spans
        _, presentation_spans = frank2015._load_mne_single_subject(subject_id, run_paths)
        span_df = pd.DataFrame(presentation_spans, columns=["start_sample", "end_sample"])
        span_df["onset"] = span_df.start_sample / frank2015.sample_rate

        # Get item onsets from annotation data.
        annot_df = frank2015._get_annotations_df(subject_id)
        annot_df = annot_df[annot_df.sentence_idx != -1]
        annot_sent_df = annot_df.groupby("sentence_idx", sort=False) \
            .agg(onset="min").reset_index()

        assert len(span_df) == len(annot_df), \
            f"Different number of sentences in signal data vs. annotations for subject {subject_id}"

        merged = pd.concat([span_df, annot_sent_df], suffixes=("", "_annots"))
        merged["onset_diff"] = merged.onset_annots - merged.onset
        diffs = np.array(merged.onset_diff)

        assert np.isclose(diffs, diffs[0]), diffs
