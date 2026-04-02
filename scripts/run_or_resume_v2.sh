#!/bin/bash
# =============================================================================
# run_or_resume_v2.sh <experiment_name> [extra_args...]
#
# Like run_or_resume.sh but loads data from a HuggingFace snapshot_download
# directory (ILSVRC/imagenet-1k) instead of a standard ImageFolder tree.
#
# Usage:
#   HF_DATA_PATH=/datasets/imagenet-1k bash scripts/run_or_resume_v2.sh simdino_vitb16_baseline
#   HF_DATA_PATH=/datasets/imagenet-1k bash scripts/run_or_resume_v2.sh efficembed_vitb16
# =============================================================================
set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <experiment_name> [extra_args...]}"
shift

CKPT_DIR="./checkpoints/${EXPERIMENT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# HF snapshot root — same dir used for both train and val splits
HF_DATA_PATH="${HF_DATA_PATH:?HF_DATA_PATH must be set to the imagenet-1k snapshot directory}"
NUM_GPUS="${NUM_GPUS:-4}"

# Detect experiment type from name
EMBED_ARGS=""
if [[ "$EXPERIMENT" == *"efficembed"* ]]; then
    EMBED_ARGS="--embed_type efficembed --sub_patch_size 4"
    echo "Detected EfficEmbed experiment"
fi

COMMON_ARGS="\
    --arch vit_base --patch_size 16 \
    --epochs 100 --batch_size_per_gpu 64 --grad_accum_steps 2 \
    --lr 0.001 --weight_decay 0.04 --weight_decay_end 0.4 \
    --warmup_epochs 10 --clip_grad 0.3 \
    --momentum_teacher 0.996 \
    --local_crops_number 10 \
    --global_crops_scale 0.4 1.0 --local_crops_scale 0.05 0.4 \
    --use_simdino True --eps 0.5 --coeff 1.0 --reduce_cov 1 --expa_type 1 \
    --use_fp16 True --compile True \
    --z_dim 256 --hidden_dim 2048 \
    --dataset_type hf_imagenet \
    --data_path ${HF_DATA_PATH} \
    --eval_data_path ${HF_DATA_PATH} \
    --output_dir ${CKPT_DIR} \
    --saveckp_freq 10 --eval_freq 10 \
    --track_wandb --wandb_project efficdino --wandb_run_name ${EXPERIMENT} \
    ${EMBED_ARGS} \
    $@"

LATEST="${CKPT_DIR}/checkpoint.pth"
if [ -f "$LATEST" ]; then
    echo "============================================"
    echo "Resuming from: ${LATEST}"
    echo "============================================"
else
    echo "============================================"
    echo "Starting fresh run: ${EXPERIMENT}"
    echo "============================================"
    mkdir -p "$CKPT_DIR"
fi

cd "$REPO_DIR/simdino"
exec torchrun --nproc_per_node=${NUM_GPUS} main_dino.py ${COMMON_ARGS}
