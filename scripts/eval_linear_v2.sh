#!/bin/bash
# =============================================================================
# Linear probe evaluation — HuggingFace ImageNet-1k (v2)
#
# Usage:
#   HF_DATA_PATH=/datasets/imagenet-1k bash scripts/eval_linear_v2.sh <checkpoint_path>
#   HF_DATA_PATH=/datasets/imagenet-1k bash scripts/eval_linear_v2.sh <checkpoint_path> --embed_type efficembed --sub_patch_size 4
# =============================================================================
set -euo pipefail

CKPT="${1:?Usage: $0 <checkpoint_path> [extra_args...]}"
shift

HF_DATA_PATH="${HF_DATA_PATH:?HF_DATA_PATH must be set to the imagenet-1k snapshot directory}"
NUM_GPUS="${NUM_GPUS:-4}"
OUTPUT_DIR="$(dirname "$CKPT")/linear_eval"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR/simdino"
torchrun --nproc_per_node=${NUM_GPUS} eval_linear.py \
    --arch vit_base --patch_size 16 \
    --pretrained_weights "$CKPT" \
    --checkpoint_key teacher \
    --dataset_type hf_imagenet \
    --data_path "$HF_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 100 --lr 0.001 --batch_size_per_gpu 256 \
    --n_last_blocks 1 --avgpool_patchtokens True \
    --num_workers 10 \
    "$@"
