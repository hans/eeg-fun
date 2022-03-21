#!/usr/bin/env nextflow

baseDir = workflow.launchDir

// Path to Broderick raw data
params.data_dir = "/om/data/public/broderick2018/Natural Speech"

eeg_dir = Channel.fromPath(params.data_dir + "/EEG")
stim_dir = Channel.fromPath(params.data_dir + "/Stimuli/Text")
//ENVELOPE_DIR = DATA_DIR / "Stimuli" / "Envelopes"

params.fulltext_path = "${baseDir}/data/texts/old-man-and-the-sea.txt"
fulltext = Channel.fromPath(params.fulltext_path)

params.language_model = "EleutherAI/gpt-neo-125M"
params.transformers_cache = "${baseDir}/transformers_cache"

// EEG processing parameters
params.filter_low = 1
params.filter_high = 8

/////////

params.outdir = "output"

process prepareStimuli {
    label "mne"

    input:
    file stim_dir from stim_dir
    file fulltext from fulltext

    output:
    file("stim_df.csv") into stim_df

    script:
"""
#!/usr/bin/env bash
TRANSFORMERS_CACHE=${params.transformers_cache} python \
    ${baseDir}/scripts/broderick2018_stimuli.py \
        ${stim_dir} \
        ${fulltext} \
        --model ${params.language_model} \
        -o stim_df.csv
"""
}

process prepareCDR {
    label "mne"

    input:
    file eeg_dir from eeg_dir
    file stim_df from stim_df

    output:
    tuple file("X.txt"), file("y.txt") into CDR_data

    script:
"""
#!/usr/bin/env python

import sys
sys.path.append("${baseDir}")

import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp"

from mfn400.adapters.broderick2018 import BroderickDatasetAdapter

dataset = BroderickDatasetAdapter("${eeg_dir}", "${stim_df}")
dataset.to_cdr("X.txt", "y.txt")
"""
}
