#!/bin/bash
# =============================================================================
# run_or_resume_dinov2.sh <experiment_name> [extra_opts...]
#
# Wrapper for simdinov2/train/train.py (DINOv2 pipeline with MCR loss).
# Auto-resumes: simdinov2 checks --output-dir for existing checkpoints.
#
# Usage:
#   HF_DATA_PATH=/datasets/imagenet-1k bash scripts/run_or_resume_dinov2.sh simdinov2_vitb16_baseline
#   HF_DATA_PATH=/datasets/imagenet-1k bash scripts/run_or_resume_dinov2.sh efficembed_vitb16
#
# Env vars:
#   HF_DATA_PATH  — required: path to HF snapshot_download root
#   NUM_GPUS      — number of GPUs per node (default: 4)
#   WANDB_API_KEY — optional: set before calling to enable WandB logging
# =============================================================================
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <experiment_name> [extra_opts...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

HF_DATA_PATH="${HF_DATA_PATH:?HF_DATA_PATH must be set to the imagenet-1k snapshot directory}"
NUM_GPUS="${NUM_GPUS:-4}"
CKPT_DIR="${REPO_DIR}/checkpoints/${EXPERIMENT}"

# Select config file based on experiment name
if [[ "$EXPERIMENT" == *"efficembed"* ]]; then
    CONFIG_FILE="${REPO_DIR}/simdinov2/configs/train/vitb16_efficembed.yaml"
    echo "Detected EfficEmbed experiment — using vitb16_efficembed.yaml"
else
    CONFIG_FILE="${REPO_DIR}/simdinov2/configs/train/vitb16_simdino.yaml"
    echo "Detected SimDINO experiment — using vitb16_simdino.yaml"
fi

mkdir -p "$CKPT_DIR"

echo "============================================"
echo "Experiment : ${EXPERIMENT}"
echo "Output dir : ${CKPT_DIR}"
echo "HF data    : ${HF_DATA_PATH}"
echo "Num GPUs   : ${NUM_GPUS}"
echo "Config     : ${CONFIG_FILE}"
echo "============================================"

cd "$REPO_DIR"
exec torchrun --nproc_per_node="${NUM_GPUS}" simdinov2/train/train.py \
    --base-config simdino_config \
    --config-file "${CONFIG_FILE}" \
    --output-dir "${CKPT_DIR}" \
    --wandb-project efficdino \
    --wandb-run-name "${EXPERIMENT}" \
    "train.dataset_path=HFImageNet:split=TRAIN:root=${HF_DATA_PATH}" \
    "$@"
