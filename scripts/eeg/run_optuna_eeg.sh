#!/bin/bash
# Optuna search on each dataset and each batch norm, as
# scripts/optuna/run_optuna_kaggle_wheat_v2.sh, with one search space for all
# (OPTUNA_CONFIG, default configs/optuna/eeg.yaml). The best trial of each study
# becomes the preset <CONFIGS>/experiment/<dataset>_<batch norm>_tuned.yaml:
#   bash run_optuna_eeg.sh
#   bash run_optuna_eeg.sh --n-trials 40 --gpu 0
#   DATASETS="bnci2014_001" BATCHNORMS="none mean_only" bash run_optuna_eeg.sh
# A dataset with subject_folds (cross-subject protocol, <dataset>_subjects) gets one
# study per fold, on the training and validation subjects of that fold only, and
# the presets <dataset>_fold<k>_<batch norm>_tuned.yaml. FOLDS restricts the folds:
#   DATASETS="bnci2014_001_subjects" FOLDS="0 1" bash run_optuna_eeg.sh
# Its batch norm is adapted to each session: classical normalization, and a fixed
# momentum (only the no-adaptation baseline K = 0 reads the running statistics).
# N_TRIALS: trials per study (default 120, 40 for the per-fold studies).
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
    n=$(sed -n 's/^subject_folds: *\([0-9]*\).*/\1/p' "$CONFIGS/dataset/$dataset.yaml")
    if [ -z "$n" ]; then folds=(""); trials=${N_TRIALS:-120}; else
        folds=(${FOLDS:-$(seq 0 $((n - 1)))}); trials=${N_TRIALS:-40}; fi
    for k in "${folds[@]}"; do
        if [ -z "$k" ]; then name="$dataset"; split=(); else
            if [ "$k" -ge "$n" ]; then echo "$dataset has folds 0 to $((n - 1)), not $k" >&2; exit 1; fi
            name="${dataset}_fold$k"; split=(--split "subject_fold_$k"); fi
        for bn in $BATCHNORMS; do
            if [ "$bn" = none ]; then fix=(batchnorm=false); else
                fix=(batchnorm_type="$bn")
                [ -n "$k" ] && fix+=(batchnorm_norm_strategy=classical batchnorm_momentum=0.01)
            fi
            out="$HERE/results_optuna/${name}_${bn}_${TIMESTAMP}"
            preset="$CONFIGS/experiment/${name}_${bn}_tuned.yaml"
            mkdir -p "$out"
            rm -f "$preset"  # never run on the preset of an older study
            python -m spdnet_training.optimize --config-dir "$CONFIGS" --config "$OPTUNA_CONFIG" \
                --dataset "$dataset" --experiment "$dataset" ${split[@]+"${split[@]}"} --fix "${fix[@]}" \
                --n-trials "$trials" --storage "sqlite:///$out/optuna_study.db" \
                --study-name "eeg_${name}_${bn}" --output-dir "$out" ${@+"$@"} 2>&1 \
                | tee "$out/optimization.log"
            if [ -f "$out/best_experiment.yaml" ]; then
                { cat "$out/best_experiment.yaml"; echo "# study: $out"; } > "$preset"
            else
                echo "No preset for $name / $bn (best trial failed): see $out/optimization.log" >&2
            fi
        done
    done
done
