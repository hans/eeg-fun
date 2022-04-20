#!/usr/bin/env nextflow

import groovy.json.JsonOutput
import nextflow.processor.TaskPath

baseDir = workflow.launchDir

// Path to Broderick raw data
params.data_dir = "/om/data/public/language-eeg/frank2015"

eeg_dir = Channel.fromPath(params.data_dir)
stim_file = Channel.fromPath(params.data_dir + "/stimuli_erp.mat")

params.language_model = "EleutherAI/gpt-neo-125M"
params.transformers_cache = "${baseDir}/transformers_cache"

params.word_freqs_path = "data/frank2015_logwordfreqs.tsv"
word_freqs = Channel.fromPath(params.word_freqs_path)

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
if (params.cdr_electrode_set == "all") {
    // Use all available electrodes in Frank dataset.
    params.cdr_response_variables = ['1', '10', '12', '14', '16', '18', '21', '22', '24',
                     '25', '26', '29', '30', '31', '33', '34', '35', '36',
                     '37', '38', '39', '40', '41', '42', '44', '45', '46',
                     '47', '48', '49', '50', '8']
} else if (params.cdr_electrode_set == "n400") {
    // Use just the electrodes specified by Frank for N400 test.
    params.cdr_response_variables = [
        "1", "14", "24", "25", "26", "29", "30", "31", "41", "42", "44", "45"
    ]
} else {
    throw new IllegalArgumentException("cdr_electrode_set must be one of `all`, `n400`")
}
// Column names for response variables in final CDR data.
cdr_response_columns = params.cdr_response_variables.collect { "el" + it }

if (!(params.cdr_response_type in ["univariate", "multivariate"])) {
    throw new IllegalArgumentException("cdr_response_type must be one of `univariate`, `multivariate`")
}
params.cdr_predictor_variables = ["rate", "surprisal", "word_freq"]
params.cdr_series_ids = "item subject"
params.cdr_history_length = 14
// Fraction of items to retain when subsetting
params.cdr_subset_item_frac = 0.3
// Fraction of subjects to retain when subsetting
params.cdr_subset_subject_frac = 1
// if true, use Frank epoch artifact annotations to NaN out regions of the
// response data.
params.cdr_drop_artifact_epochs = false

/////////

params.outdir = "${baseDir}/output/frank2015"

/////////

// Duplicate channels for multiple processing streams
eeg_dir.into { eeg_dir_for_erp; eeg_dir_for_cdr }
stim_file.into { stim_file_for_prep; stim_file_for_repro }

/////////

// Header for Python scripts which makes mfn400 import available
PYTHON_HEADER = """
#!/usr/bin/env python

import sys
sys.path.append("${baseDir}")
import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
"""

/////////

process prepareStimuli {
    label "mne"
    publishDir "${params.outdir}"

    input:
    file stim_file from stim_file_for_prep
    file word_freqs from word_freqs

    output:
    file("stim_df.csv") into stim_df

    script:
"""
#!/usr/bin/env bash
export PYTHONPATH="${baseDir}"
export NUMBA_CACHE_DIR="/tmp"
TRANSFORMERS_CACHE=${params.transformers_cache} python \
    ${baseDir}/scripts/frank2015_stimuli.py \
        ${stim_file} \
        --model ${params.language_model} \
        --word-freqs-path ${word_freqs} \
        -o stim_df.csv
"""
}

stim_df.into { stim_df_for_erp; stim_df_for_cdr }

/**
 * Join raw EEG data with stimulus data and convert to CDR format.
 */
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
${PYTHON_HEADER}

from mfn400.adapters.frank2015 import FrankDatasetAdapter

dataset = FrankDatasetAdapter("${eeg_dir}", "${stim_df}")
dataset.to_cdr("X.txt", "y.txt")
"""
}

/**
 * Preprocess stimulus and EEG response data:
 *
 * 1. Prepare response variable (electrode subsetting; spatial/temporal averaging)
 * 2. Zero out time spans of response variable which are known to have
 *    substantial artifacts
 * 3. Adjust time axis (zero out at start of each item)
 * 4. Create train/dev/test split
 */
process simplifyCDR {
    label "mne"

    when:
    params.mode == "cdr"

    input:
    tuple file(X), file(y) from CDR_data

    output:
    tuple file("X_simp.txt"),
          file("y_simp.train.txt"), file("y_simp.dev.txt"), file("y_simp.test.txt") into CDR_data_simple

    script:

    electrodes_str = JsonOutput.toJson(params.cdr_response_variables)
    electrode_columns_str = JsonOutput.toJson(cdr_response_columns)
    electrodes_arr = "[" + params.cdr_response_variables.join(",") + "]"
    switch_univariate = params.cdr_response_type == "univariate" ? "True" : "False"

    switch_zero = params.cdr_drop_artifact_epochs ? "True" : "False"
"""
${PYTHON_HEADER}

import numpy as np
import pandas as pd

ELECTRODES = ${electrodes_str}
ELECTRODES = [str(el) for el in ELECTRODES]
ELECTRODE_COLS = ${electrode_columns_str}

X = pd.read_csv("${X}", sep=" ")
y = pd.read_csv("${y}", sep=" ")

y = y.rename(columns=dict(zip(ELECTRODES, ELECTRODE_COLS)))

if ${switch_univariate}:
    y["mean_response"] = y[ELECTRODE_COLS].mean(axis=1)
    y = y.drop(columns=ELECTRODE_COLS)
    response_indexer = ["mean_response"]
else:
    response_indexer = ELECTRODE_COLS

# Drop response regions corresponding to epochs in the Frank preprocessed
# data that were marked as containing artifacts.
if ${switch_zero}:
    bad_epochs = X[X.artifact == 1]
    bad_epochs["drop_time_min"] = bad_epochs.time + ${params.erp_epoch_window_left}
    bad_epochs["drop_time_max"] = bad_epochs.time + ${params.erp_epoch_window_right}

    # TODO is there a clean vectorized solution?
    # Indexing on subject, item; then using custom float range on response tbl
    for _, row in bad_epochs.iterrows():
        y.loc[(y.subject == row.subject) &
              (y.time >= row.drop_time_min) &
              (y.time < row.drop_time_max), response_indexer] = np.nan

    n_nanned = y[response_indexer[0]].isna().sum()
    print(f"NaN'ed out {len(bad_epochs)} -- that's {n_nanned} response rows "
        f"({n_nanned / len(y) * 100}%)")

# Zero out clock at the start of each item.
item_times = pd.DataFrame(X.groupby(["subject", "item"]).time.min())
item_times["y_time"] = y.groupby(["subject", "item"]).time.min()
item_times["min_time"] = item_times.min(axis=1)
X.time -= X.merge(item_times, how="left", left_on=["subject", "item"], right_index=True).min_time
y.time -= y.merge(item_times, how="left", left_on=["subject", "item"], right_index=True).min_time

# Compute train/dev/test splits.
X["modulus"] = (X["item"] + X["subject"]) % 4
y["modulus"] = (y["item"] + y["subject"]) % 4
mapping = {0: "train", 1: "train", 2: "dev", 3: "test"}
X["target"] = X.modulus.map(mapping)
y["target"] = y.modulus.map(mapping)

# Only y should be partitioned in separate files, per CDR design.
X.to_csv("X_simp.txt", sep=" ", index=False)
for target in set(mapping.values()):
    y[y.target == target].to_csv(f"y_simp.{target}.txt", sep=" ", index=False,
                                 float_format="%.4f")
"""
}

CDR_data_simple.into { CDR_data_simple_for_train; CDR_data_simple_for_subset }

def makeCDRInvocation(TaskPath X, TaskPath y_train, TaskPath y_dev, TaskPath y_test) {
    response_expr = params.cdr_response_type == "univariate"
        ? "mean_response"
        : cdr_response_columns.join(" + ")
    predictor_expr = params.cdr_predictor_variables.join(" + ")
    formula = "${response_expr} ~ C(${predictor_expr}, NN()) + (C(${predictor_expr}, NN(ran=T)) | subject)"

    """
    #!/usr/bin/env bash

    export X="${X}"
    export y_train="${y_train}"
    export y_dev="${y_dev}"
    export y_test="${y_test}"
    export outdir="${params.outdir}"
    export history_length="${params.cdr_history_length}"
    export series_ids="${params.cdr_series_ids}"
    export formula="${formula}"
    export model_name="CDR_full"

    envsubst < ${baseDir}/cdr_config_template.ini > cdr.ini

    python -m cdr.bin.train cdr.ini
    """
}

process runCDR {
    label "cdr"
    publishDir "${params.outdir}"

    when:
    params.mode == "cdr" && !params.cdr_subset

    input:
    tuple file(X),
          file(y_train), file(y_dev), file(y_test) from CDR_data_simple_for_train

    output:
    file("CDR_full") into CDR_model

    script:
    makeCDRInvocation(X, y_train, y_dev, y_test)
}

/**
 * Prepare subsetted CDR data for quick exploratory fits, etc.
 */
process subsetCDR {
    label "mne"
    publishDir "${params.outdir}/subset"

    when:
    params.cdr_subset

    input:
    tuple file(X),
          file(y_train), file(y_dev), file(y_test) from CDR_data_simple_for_subset

    output:
    tuple file("X_subset.txt"),
          file("y_subset.train.txt"), file("y_subset.dev.txt"), file("y_subset.test.txt") into CDR_data_subset

    script:
"""
${PYTHON_HEADER}

import numpy as np
import pandas as pd

X = pd.read_csv("${X}", sep=" ")
y_train, y_dev, y_test = [pd.read_csv(f, sep=" ") for f in ["${y_train}",
                                                            "${y_dev}",
                                                            "${y_test}"]]

keep_subject_frac = ${params.cdr_subset_subject_frac}
keep_item_frac = ${params.cdr_subset_item_frac}

subjects, items = X.subject.unique(), X.item.unique()
n_subjects = int(np.floor(keep_subject_frac * len(subjects)))
n_items = int(np.floor(keep_item_frac * len(items)))

keep_subjects = np.random.choice(subjects, n_subjects, replace=False)
keep_items = np.random.choice(items, n_items, replace=False)

for dataset in [X, y_train, y_dev, y_test]:
    dataset.drop(dataset[~(dataset.subject.isin(keep_subjects)
                           & dataset.item.isin(keep_items))].index,
                 inplace=True)

X.to_csv("X_subset.txt", sep=" ", index=False)
for y_dataset in [y_train, y_dev, y_test]:
    label = y_dataset.iloc[0].target
    y_dataset.to_csv(f"y_subset.{label}.txt", sep=" ", index=False,
                     float_format="%.4f")
"""
}

process runCDRSubset {
    label "cdr"
    publishDir "${params.outdir}/subset"

    when:
    params.mode == "cdr"

    input:
    tuple file(X),
          file(y_train), file(y_dev), file(y_test) from CDR_data_subset

    output:
    file("CDR_subset") into CDR_model_subset

    script:
    makeCDRInvocation(X, y_train, y_dev, y_test)
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
