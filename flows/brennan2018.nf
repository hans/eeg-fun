#!/usr/bin/env nextflow

baseDir = workflow.launchDir

// Path to Broderick raw data
params.data_dir = "/om/data/public/language-eeg/brennan2018-v2"

eeg_dir = Channel.fromPath(params.data_dir)
stim_df = Channel.fromPath("${params.data_dir}/stimuli/AliceChapterOne-EEG.csv")
raw_text = Channel.fromPath("${baseDir}/data/texts/alice-ch1.txt")

word_freqs = Channel.fromPath("${baseDir}/data/wikitext-2_train_vocab.txt")
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

process prepareStimuli {
    conda "/home/jgauthie/.conda/envs/huggingface"
    publishDir "${params.outdir}"

    input:
    file stim_df from stim_df
    file raw_text from raw_text
    file word_freqs from word_freqs

    output:
    file("stim_df_with_surprisals.csv") into stim_df_with_surprisals

    script:
"""
#!/usr/bin/env bash
TRANSFORMERS_CACHE=${params.transformers_cache} python \
    ${baseDir}/scripts/brennan2018_stimuli.py \
    ${stim_df} ${raw_text} \
    --model ${params.language_model} \\
    --word-freqs-path ${word_freqs} \\
    -o stim_df_with_surprisals.csv
"""
}

process prepareERP {
    label "mne"
    publishDir "${params.outdir}/erp"

    when:
    params.mode == "erp"

    input:
    file eeg_dir from eeg_dir_for_erp
    file stim_df from stim_df_with_surprisals

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
