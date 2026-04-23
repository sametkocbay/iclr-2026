#!/bin/bash
#SBATCH --job-name=stmno_2_28k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --qos=mcml
#SBATCH --partition=mcml-dgx-a100-40x8,mcml-hgx-a100-80x4,mcml-hgx-h100-94x4
#SBATCH --output=/dss/mcmlscratch/08/di97hih/logs/%x-%j.out
#SBATCH --error=/dss/mcmlscratch/08/di97hih/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="/dss/dsshome1/08/di97hih/iclr_develop"
SCRATCH_ROOT="${MCMLSCRATCH:-/dss/mcmlscratch/08/di97hih}"
DATASET_DIR="${PROJECT_ROOT}/dataset_huggingface/warped-ifw"
RUN_PREFIX="${RUN_PREFIX:-spatiotemporal_mno_seed118}"
JOB_NAME="${SLURM_JOB_NAME:-stmno_2_28k}"
RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_${JOB_NAME}_${RUN_ID}}"
RUNS_DIR="${SCRATCH_ROOT}/outputs/runs"
CHECKPOINT_PATH="${RUNS_DIR}/${RUN_NAME}/checkpoints/best.pt"
LOG_DIR="${SCRATCH_ROOT}/logs"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
NUM_POINTS="${NUM_POINTS:-28000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
ENABLE_RANDOM_SUBSAMPLE="${ENABLE_RANDOM_SUBSAMPLE:-1}"
BOUNDARY_POINT_FRACTION="${BOUNDARY_POINT_FRACTION:-0.0}"
EPOCHS="${EPOCHS:-200}"
ENABLE_ACTIVATION_CHECKPOINTING="${ENABLE_ACTIVATION_CHECKPOINTING:-0}"
ENABLE_PRECOMPUTE_KNN="${ENABLE_PRECOMPUTE_KNN:-1}"
KNN_CACHE_DIR="${KNN_CACHE_DIR:-${SCRATCH_ROOT}/outputs/knn_cache}"
KNN_INCLUDE_NEIGHBORS="${KNN_INCLUDE_NEIGHBORS:-0}"
ENABLE_COMPILE_MODEL="${ENABLE_COMPILE_MODEL:-0}"
COMPILE_CACHE_ROOT="${COMPILE_CACHE_ROOT:-${SCRATCH_ROOT}/outputs/compile_cache}"
RESUME_FROM="${RESUME_FROM:-}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/outputs/wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-iclr_develop}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"

JOB_CACHE_ID="${SLURM_JOB_ID:-manual}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${COMPILE_CACHE_ROOT}/torchinductor/${JOB_CACHE_ID}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${COMPILE_CACHE_ROOT}/triton/${JOB_CACHE_ID}}"

mkdir -p "${LOG_DIR}" "${RUNS_DIR}" "$(dirname "${CHECKPOINT_PATH}")" "${WANDB_DIR}" "${KNN_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Create the virtualenv first or update PYTHON_BIN in $(basename "$0")." >&2
    exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "Dataset directory not found: ${DATASET_DIR}" >&2
    exit 1
fi

if [[ "${ENABLE_WANDB}" == "1" ]] && ! "${PYTHON_BIN}" -c "import wandb" >/dev/null 2>&1; then
    echo "W&B logging is enabled but wandb is not installed in ${PYTHON_BIN}." >&2
    echo "Install it with: ${PYTHON_BIN} -m pip install wandb" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Project root: ${PROJECT_ROOT}"
echo "Dataset dir: ${DATASET_DIR}"
echo "Run prefix: ${RUN_PREFIX}"
echo "Job name: ${JOB_NAME}"
echo "Run id: ${RUN_ID}"
echo "Run name: ${RUN_NAME}"
echo "Runs dir: ${RUNS_DIR}"
echo "Checkpoint path: ${CHECKPOINT_PATH}"
echo "Model: spatiotemporal_mno"
echo "Num points: ${NUM_POINTS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Grad accumulation: ${GRAD_ACCUM_STEPS}"
echo "Random subsample: ${ENABLE_RANDOM_SUBSAMPLE}"
echo "Boundary point fraction: ${BOUNDARY_POINT_FRACTION}"
echo "Epochs: ${EPOCHS}"
echo "Activation checkpointing: ${ENABLE_ACTIVATION_CHECKPOINTING}"
echo "Precompute kNN: ${ENABLE_PRECOMPUTE_KNN}"
echo "kNN cache dir: ${KNN_CACHE_DIR}"
echo "kNN include crop neighbors: ${KNN_INCLUDE_NEIGHBORS}"
echo "Compile model: ${ENABLE_COMPILE_MODEL}"
echo "TorchInductor cache dir: ${TORCHINDUCTOR_CACHE_DIR}"
echo "Triton cache dir: ${TRITON_CACHE_DIR}"
echo "Resume from: ${RESUME_FROM:-unset}"
echo "Enable W&B: ${ENABLE_WANDB}"
echo "W&B dir: ${WANDB_DIR}"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B run name: ${WANDB_RUN_NAME}"
echo "W&B mode: ${WANDB_MODE}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
fi

TRAIN_ARGS=(
    --model-name spatiotemporal_mno
    --config "${PROJECT_ROOT}/config/spatiotemporal_mno.yaml"
    --device cuda
    --checkpoint-path "${CHECKPOINT_PATH}"
    --dataset-dir "${DATASET_DIR}"
    --num-points "${NUM_POINTS}"
    --val-fraction 0.1
    --test-fraction 0.1
    --boundary-point-fraction "${BOUNDARY_POINT_FRACTION}"
    --latent-dim 128
    --num-modes 256
    --num-blocks 4
    --num-heads 8
    --k 16
    --batch-size "${BATCH_SIZE}"
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}"
    --lr 0.001
    --weight-decay 0.0001
    --epochs "${EPOCHS}"
    --seed 118
    --loss-fn rl2
    --lr-scheduler auto
    --use-amp
    --runs-dir "${RUNS_DIR}"
    --run-name "${RUN_NAME}"
)

if [[ "${ENABLE_COMPILE_MODEL}" == "1" ]]; then
    TRAIN_ARGS+=(--compile-model)
else
    TRAIN_ARGS+=(--no-compile-model)
fi

if [[ "${ENABLE_RANDOM_SUBSAMPLE}" == "1" ]]; then
    TRAIN_ARGS+=(--random-subsample)
fi

if [[ "${ENABLE_ACTIVATION_CHECKPOINTING}" == "1" ]]; then
    TRAIN_ARGS+=(--activation-checkpointing)
fi

if [[ "${ENABLE_PRECOMPUTE_KNN}" == "1" ]]; then
    TRAIN_ARGS+=(
        --precompute-knn
        --knn-cache-dir "${KNN_CACHE_DIR}"
    )
fi

if [[ "${KNN_INCLUDE_NEIGHBORS}" == "1" ]]; then
    TRAIN_ARGS+=(--knn-include-neighbors)
fi

if [[ -n "${RESUME_FROM}" ]]; then
    TRAIN_ARGS+=(--resume-from "${RESUME_FROM}")
fi

if [[ "${ENABLE_WANDB}" == "1" ]]; then
    TRAIN_ARGS+=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT}"
        --wandb-run-name "${WANDB_RUN_NAME}"
        --wandb-dir "${WANDB_DIR}"
        --wandb-mode "${WANDB_MODE}"
    )
fi

"${PYTHON_BIN}" scripts/train.py "${TRAIN_ARGS[@]}"
