#!/bin/bash
# Optuna search on each dataset and each batch norm, as
# scripts/optuna/run_optuna_kaggle_wheat_v2.sh, with one search space for all
# (OPTUNA_CONFIG, default configs/optuna/eeg.yaml). The best trial of each study
# becomes the preset <CONFIGS>/experiment/<dataset>_<batch norm>_tuned.yaml:
#   bash run_optuna_eeg.sh
#   bash run_optuna_eeg.sh --n-trials 40 --gpu 0
#   DATASETS="bnci2014_001" BATCHNORMS="none mean_only" bash run_optuna_eeg.sh
# BATCHNORMS: "none" (no batch norm) or values of model.batchnorm_type.
# CONFIGS: config directory with dataset/, experiment/ and optuna/ (default: configs/ here).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS="${CONFIGS:-$HERE/configs}"
OPTUNA_CONFIG="${OPTUNA_CONFIG:-eeg}"
DATASETS="${DATASETS:-bnci2014_001 weibo2014 physionetmi cho2017 schirrmeister2017}"
BATCHNORMS="${BATCHNORMS:-none mean_only mean_var_scalar}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
for dataset in $DATASETS; do
    for bn in $BATCHNORMS; do
        if [ "$bn" = none ]; then fix=(batchnorm=false); else fix=(batchnorm_type="$bn"); fi
        out="$HERE/results_optuna/${dataset}_${bn}_${TIMESTAMP}"
        mkdir -p "$out"
        python -m spdnet_training.optimize --config-dir "$CONFIGS" --config "$OPTUNA_CONFIG" \
            --dataset "$dataset" --experiment "$dataset" --fix "${fix[@]}" --n-trials 120 \
            --storage "sqlite:///$out/optuna_study.db" --study-name "eeg_${dataset}_${bn}" \
            --output-dir "$out" "$@" 2>&1 | tee "$out/optimization.log"
        if [ -f "$out/best_experiment.yaml" ]; then
            cp "$out/best_experiment.yaml" "$CONFIGS/experiment/${dataset}_${bn}_tuned.yaml"
        else
            echo "No preset for $dataset / $bn (best trial failed): see $out/optimization.log" >&2
        fi
    done
done
