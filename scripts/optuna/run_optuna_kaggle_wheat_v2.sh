#!/bin/bash
# =============================================================================
# Optuna Optimization v2: Kaggle Wheat Disease -- Joint SGD/Adam + Depth
# =============================================================================
# Based on v1 results (best=0.8617). New in v2:
#   - Joint SGD/Adam optimizer search
#   - 3rd hidden layer (-1 = skip) for depth testing
#   - Strictly decreasing layer sizes
#   - 120 trials, adjusted ranges
#
# Usage:
#   bash run_optuna_kaggle_wheat_v2.sh              # run with defaults
#   bash run_optuna_kaggle_wheat_v2.sh --resume      # resume interrupted study
#   bash run_optuna_kaggle_wheat_v2.sh --gpu 1       # pin to GPU 1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTPUT_DIR="${SCRIPT_DIR}/results_kaggle_wheat_v2_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

# Activate venv
if [[ -f "${PROJECT_ROOT}/.venv_spdnet-bench/bin/activate" ]]; then
    source "${PROJECT_ROOT}/.venv_spdnet-bench/bin/activate"
fi

# Forward extra CLI flags (e.g. --resume, --gpu 0)
EXTRA_ARGS=("$@")

echo "========================================"
echo " Optuna Optimization v2"
echo " Dataset   : Kaggle Wheat"
echo " Optimizer : SGD + Adam (joint search)"
echo " Config    : configs/optuna/kaggle_wheat_v2.yaml"
echo " Trials    : 120"
echo " Output    : ${OUTPUT_DIR}"
echo " Extra args: ${EXTRA_ARGS[*]:-none}"
echo "========================================"

python -m spdnet_training.optimize \
    --config kaggle_wheat_v2 \
    --experiment kaggle_wheat_sgd_batchnorm \
    --n-trials 120 \
    --storage "sqlite:///${OUTPUT_DIR}/optuna_study.db" \
    --study-name kaggle_wheat_optuna_v2 \
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
echo "   optuna-dashboard sqlite:////${OUTPUT_DIR}/optuna_study.db --port 8080 --host 0.0.0.0"
echo "========================================"
