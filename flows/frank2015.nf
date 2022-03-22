#!/usr/bin/env nextflow

baseDir = workflow.launchDir

// Path to Broderick raw data
params.data_dir = "/om/data/public/language-eeg/frank2015"

eeg_dir = Channel.fromPath(params.data_dir)
stim_file = Channel.fromPath(params.data_dir + "/stimuli_erp.mat")

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

params.outdir = "${baseDir}/output"

process prepareStimuli {
    label "mne"
    publishDir "${params.outdir}"

    input:
    file stim_file from stim_file

    output:
    file("stim_df.csv") into stim_df

    script:
"""
#!/usr/bin/env bash
export PYTHONPATH="${baseDir}"
TRANSFORMERS_CACHE=${params.transformers_cache} python \
    ${baseDir}/scripts/frank2015_stimuli.py \
        ${stim_file} \
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

from mfn400.adapters.frank2015 import FrankDatasetAdapter

dataset = FrankDatasetAdapter("${eeg_dir}", "${stim_df}")
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
