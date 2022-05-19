#!/usr/bin/env nextflow

baseDir = workflow.launchDir

// Path to Broderick raw data
params.data_dir = "/om/data/public/language-eeg/brennan2018-v2"

eeg_dir = Channel.fromPath(params.data_dir)

params.language_model = "EleutherAI/gpt-neo-125M"
params.transformers_cache = "${baseDir}/transformers_cache"

/**
 * Specify the analysis to carry out. One of "erp", ...
 */
params.mode = "erp"

// ERP parameters
params.erp_epoch_window_left = -0.1
params.erp_epoch_window_right = 0.924
params.erp_test_window_left = 0.3
params.erp_test_window_right = 0.5

/////////

params.outdir = "${baseDir}/output/brennan2018"

/////////

eeg_dir.into { eeg_dir_for_erp }

/////////

/*
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
TRANSFORMERS_CACHE=${params.transformers_cache} python \
    ${baseDir}/scripts/frank2015_stimuli.py \
        ${stim_file} \
        --model ${params.language_model} \
        --word-freqs-path ${word_freqs} \
        -o stim_df.csv
"""
}*/

stim_df.into { stim_df_for_erp }

process prepareERP {
    label "mne"
    publishDir "${params.outdir}/erp"

    when:
    params.mode == "erp"

    input:
    file eeg_dir from eeg_dir_for_erp
    file stim_df from stim_df_for_erp

    output:
    file "erp_full.csv" into erp_df

    script:
"""
#!/usr/bin/env bash

export PYTHONPATH="${baseDir}"
export NUMBA_CACHE_DIR=/tmp

# TODO pass epoching etc. parameters

python ${baseDir}/scripts/brennan2018_erp.py \
    ${eeg_dir} \
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
    file "Naturalistic-N400-Brennan.nb.html"

    script:
"""
#!/usr/bin/env bash

#cd ${baseDir}/notebooks
Rscript -e "rmarkdown::render('${baseDir}/notebooks/Naturalistic N400 Brennan.Rmd', params=list(file='${erp_df}'))"
"""
}
