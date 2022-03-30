#!/usr/bin/env nextflow

baseDir = workflow.launchDir

// Path to Broderick raw data
params.data_dir = "/om/data/public/language-eeg/frank2015"

eeg_dir = Channel.fromPath(params.data_dir)
stim_file = Channel.fromPath(params.data_dir + "/stimuli_erp.mat")

params.language_model = "EleutherAI/gpt-neo-125M"
params.transformers_cache = "${baseDir}/transformers_cache"

/**
 * Specify the analysis to carry out. One of "erp", "cdr"
 */
params.mode = "cdr"

// ERP parameters
params.erp_epoch_window_left = -0.1
params.erp_epoch_window_right = 0.924
params.erp_test_window_left = 0.3
params.erp_test_window_right = 0.5

// CDR parameters
params.cdr_response_variables = [
                      '1', '10', '12', '14', '16', '18', '21', '22', '24',
                     '25', '26', '29', '30', '31', '33', '34', '35', '36',
                     '37', '38', '39', '40', '41', '42', '44', '45', '46',
                     '47', '48', '49', '50', '8']
params.cdr_predictor_variables = ["surprisal"]
params.cdr_series_ids = "item subject"

/////////

params.outdir = "${baseDir}/output/frank2015"

/////////

// Duplicate channels for the two processing streams
eeg_dir.into { eeg_dir_for_erp; eeg_dir_for_cdr }
stim_file.into { stim_file_for_prep; stim_file_for_repro }

/////////

process prepareStimuli {
    label "mne"
    publishDir "${params.outdir}"

    input:
    file stim_file from stim_file_for_prep

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

stim_df.into { stim_df_for_erp; stim_df_for_cdr }

process prepareCDR {
    label "mne"

    when:
    params.mode == "cdr"

    input:
    file eeg_dir from eeg_dir_for_cdr
    file stim_df from stim_df_for_cdr

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
dataset.to_cdr("X.txt", "y.txt")
"""
}

/**
 * Average across electrodes of interest.
 */
process simplifyCDR {
    label "mne"

    when:
    params.mode == "cdr"

    input:
    tuple file(X), file(y) from CDR_data

    output:
    tuple file("X_simp.txt"), file("y_simp.txt") into CDR_data_simple

    script:

    electrodes_arr = "[" + params.cdr_response_variables.join(",") + "]"
"""
#!/usr/bin/env python

import pandas as pd

ELECTRODES = ${electrodes_arr}
ELECTRODES = [str(el) for el in ELECTRODES]

X = pd.read_csv("${X}", sep=" ")

y = pd.read_csv("${y}", sep=" ")
y["mean_response"] = y[ELECTRODES].mean(axis=1)
y = y.drop(columns=ELECTRODES)

# Zero out clock at the start of each item.
item_times = pd.DataFrame(X.groupby(["subject", "item"]).time.min())
item_times["y_time"] = y.groupby(["subject", "item"]).time.min()
item_times["min_time"] = item_times.min(axis=1)
X.time -= X.merge(item_times, how="left", left_on=["subject", "item"], right_index=True).min_time
y.time -= y.merge(item_times, how="left", left_on=["subject", "item"], right_index=True).min_time

X.to_csv("X_simp.txt", sep=" ", index=False)
y.to_csv("y_simp.txt", sep=" ", index=False,
         float_format="%.4f")
"""
}

process runCDR {
    label "cdr"
    publishDir "${params.outdir}"

    when:
    params.mode == "cdr"

    input:
    tuple file(X), file(y) from CDR_data_simple

    script:
    response_expr = "mean_response"  // params.cdr_response_variables.join(" + ")
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

process prepareERPControl {
    label "mne"

    when:
    params.mode == "erp"

    input:
    file stim_raw from stim_file_for_repro

    output:
    file "erp.csv" into erp_control_df

    script:
"""
#!/usr/bin/env bash

python ${baseDir}/scripts/frank2015_erp_premade.py \
    ${stim_raw} erp.csv
"""
}

process prepareERP {
    label "mne"
    publishDir "${params.outdir}/erp"

    when:
    params.mode == "erp"

    input:
    file eeg_dir from eeg_dir_for_erp
    file stim_df from stim_df_for_erp
    file erp_control_df from erp_control_df

    output:
    file "erp_full.csv" into erp_df
    file "n400_comparison.png"

    script:
"""
#!/usr/bin/env bash

export PYTHONPATH="${baseDir}"
export NUMBA_CACHE_DIR=/tmp

# TODO pass epoching etc. parameters

python ${baseDir}/scripts/frank2015_erp_repro.py \
    ${eeg_dir} ${stim_df} \
    -r ${erp_control_df} \
    -o erp_full.csv
"""
}

process runERP {
    label "r"
    publishDir "${params.outdir}/erp"

    when:
    params.mode == "erp"

    input:
    file erp_df from erp_df

    output:
    file "Naturalistic-N400-Frank.nb.html"

    script:
"""
#!/usr/bin/env bash

#cd ${baseDir}/notebooks
Rscript -e "rmarkdown::render('${baseDir}/notebooks/Naturalistic N400 Frank.Rmd', params=list(file='${erp_df}'))"
"""
}
