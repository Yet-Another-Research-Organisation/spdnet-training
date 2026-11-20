#!/bin/bash
# Runs spdnet-train on the preset of each dataset, with the same extra overrides:
#   bash run_eeg.sh
#   bash run_eeg.sh --multirun seed=0,1,2,3,4 logging.rich_progress=false
#   DATASETS="bnci2014_001 cho2017" bash run_eeg.sh
# With the presets written by run_optuna_eeg.sh, one run per dataset and batch norm:
#   BATCHNORMS="none mean_only mean_var_scalar" bash run_eeg.sh --multirun seed=0,1,2,3,4
# CONFIGS: config directory with dataset/ and experiment/ (default: configs/ here).
# Options such as --multirun come first.
set -euo pipefail
CONFIGS="${CONFIGS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/configs}"
DATASETS="${DATASETS:-bnci2014_001 weibo2014 physionetmi cho2017 schirrmeister2017}"
for dataset in $DATASETS; do
    if [ -z "${BATCHNORMS:-}" ]; then
        spdnet-train --config-dir "$CONFIGS" "$@" experiment="$dataset"
        continue
    fi
    for bn in $BATCHNORMS; do
        if [ ! -f "$CONFIGS/experiment/${dataset}_${bn}_tuned.yaml" ]; then
            echo "No preset ${dataset}_${bn}_tuned (study missing or failed): skipped" >&2
            continue
        fi
        spdnet-train --config-dir "$CONFIGS" "$@" experiment="${dataset}_${bn}_tuned"
    done
done
