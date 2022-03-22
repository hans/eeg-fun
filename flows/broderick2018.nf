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

// CDR parameters
params.cdr_response_variables = (1..128).collect({"V" + it})
params.cdr_predictor_variables = ["surprisal"]
params.cdr_series_ids = "item subject"

/////////

params.outdir = "${baseDir}/output/broderick2018"

process prepareStimuli {
    label "mne"
    publishDir "${params.outdir}"

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
dataset.to_cdr("X.txt", "y.txt",
               filter_window=(${params.filter_low},
                              ${params.filter_high}))
"""
}

process runCDR {
    label "cdr"
    publishDir "${params.outdir}"

    input:
    tuple file(X), file(y) from CDR_data

    script:
    response_expr = params.cdr_response_variables.join(" + ")
    predictor_expr = params.cdr_predictor_variables.join(" + ")
    formula = "${response_expr} ~ C(${predictor_expr}, NN())"

"""
#!/usr/bin/env bash

export X_train="${X}"
export y_train="${y}"
export outdir="${params.outdir}"
export series_ids="${params.cdr_series_ids}"
export formula="${formula}"

envsubst < ${baseDir}/cdr_config_template.ini > cdr.ini

python -m cdr.bin.train cdr.ini
"""
}
