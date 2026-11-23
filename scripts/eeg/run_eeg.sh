#!/bin/bash
# Runs spdnet-train on the preset of each dataset, with the same extra overrides:
#   bash run_eeg.sh
#   bash run_eeg.sh --multirun seed=0,1,2,3,4 logging.rich_progress=false
#   DATASETS="bnci2014_001 cho2017" bash run_eeg.sh
# With the presets written by run_optuna_eeg.sh, one run per dataset and batch norm:
#   BATCHNORMS="none mean_only mean_var_scalar" bash run_eeg.sh --multirun seed=0,1,2,3,4
# A dataset with subject_folds (cross-subject protocol, <dataset>_subjects) runs every
# fold, each with its own preset <dataset>_fold<k>_<batch norm>_tuned.yaml. FOLDS
# restricts the folds. Each run writes target_metrics.json (test accuracy for each
# calibration size K):
#   DATASETS="bnci2014_001_subjects" BATCHNORMS="mean_only" bash run_eeg.sh --multirun seed=0,1
# CONFIGS: config directory with dataset/ and experiment/ (default: configs/ here).
# Options such as --multirun come first.
set -euo pipefail
CONFIGS="${CONFIGS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/configs}"
DATASETS="${DATASETS:-bnci2014_001 weibo2014 physionetmi cho2017 schirrmeister2017}"
for dataset in $DATASETS; do
    n=$(sed -n 's/^subject_folds: *\([0-9]*\).*/\1/p' "$CONFIGS/dataset/$dataset.yaml")
    if [ -z "$n" ]; then folds=(""); else folds=(${FOLDS:-$(seq 0 $((n - 1)))}); fi
    for k in "${folds[@]}"; do
        if [ -z "$k" ]; then name="$dataset"; split=(); else
            if [ "$k" -ge "$n" ]; then echo "$dataset has folds 0 to $((n - 1)), not $k" >&2; exit 1; fi
            name="${dataset}_fold$k"; split=(dataset.split_strategy="subject_fold_$k"); fi
        if [ -z "${BATCHNORMS:-}" ]; then
            spdnet-train --config-dir "$CONFIGS" ${@+"$@"} experiment="$dataset" ${split[@]+"${split[@]}"}
            continue
        fi
        for bn in $BATCHNORMS; do
            if [ ! -f "$CONFIGS/experiment/${name}_${bn}_tuned.yaml" ]; then
                echo "No preset ${name}_${bn}_tuned (study missing or failed): skipped" >&2
                continue
            fi
            spdnet-train --config-dir "$CONFIGS" ${@+"$@"} experiment="${name}_${bn}_tuned"
        done
    done
done
