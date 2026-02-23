#!/bin/bash
# =============================================================================
# Optuna Optimization: Kaggle Wheat Disease -- SGD + Warmup Plateau
# =============================================================================
# Jointly optimizes ALL hyperparameters using Bayesian optimization (TPE).
# Each trial runs 3 seeds; the objective is the mean test loss (minimize).
#
# Replaces the sequential grid-search pipeline (01->07).
#
# Usage:
#   bash run_optuna_kaggle_wheat_sgd.sh           # run with defaults
#   bash run_optuna_kaggle_wheat_sgd.sh --resume   # resume interrupted study
#   bash run_optuna_kaggle_wheat_sgd.sh --gpu 1    # pin to GPU 1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTPUT_DIR="${SCRIPT_DIR}/results_kaggle_wheat_sgd_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

# Activate venv
if [[ -f "${PROJECT_ROOT}/.venv_spdnet-bench/bin/activate" ]]; then
    source "${PROJECT_ROOT}/.venv_spdnet-bench/bin/activate"
fi

# Forward extra CLI flags (e.g. --resume, --gpu 0)
EXTRA_ARGS=("$@")

echo "========================================"
echo " Optuna Optimization"
echo " Dataset   : Kaggle Wheat"
echo " Optimizer : SGD + Warmup Plateau"
echo " Config    : configs/optuna/kaggle_wheat_sgd.yaml"
echo " Output    : ${OUTPUT_DIR}"
echo " Extra args: ${EXTRA_ARGS[*]:-none}"
echo "========================================"

python -m spdnet_training.optimize \
    --config kaggle_wheat_sgd \
    --experiment kaggle_wheat_sgd_batchnorm \
    --n-trials 60 \
    --storage "sqlite:///${OUTPUT_DIR}/optuna_study.db" \
    --study-name kaggle_wheat_sgd_optuna \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/optimization.log"

echo ""
echo "========================================"
echo " Optimization complete!"
echo " Results : ${OUTPUT_DIR}"
echo " DB      : ${OUTPUT_DIR}/optuna_study.db"
echo ""
echo " Inspect interactively:"
echo "   optuna-dashboard sqlite:///${OUTPUT_DIR}/optuna_study.db"
echo "========================================"
