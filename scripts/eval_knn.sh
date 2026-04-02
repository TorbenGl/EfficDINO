#!/bin/bash
# =============================================================================
# k-NN evaluation on ImageNet val set
#
# Usage:
#   bash scripts/eval_knn.sh <checkpoint_path> [--embed_type efficembed]
#
# Examples:
#   bash scripts/eval_knn.sh ./checkpoints/simdino_vitb16_baseline/checkpoint_best.pth
#   bash scripts/eval_knn.sh ./checkpoints/efficembed_vitb16/checkpoint_best.pth --embed_type efficembed --sub_patch_size 4
# =============================================================================
set -euo pipefail

CKPT="${1:?Usage: $0 <checkpoint_path> [extra_args...]}"
shift

DATA_PATH="${DATA_PATH:-/datasets/imagenet}"
NUM_GPUS="${NUM_GPUS:-4}"
DUMP_DIR="$(dirname "$CKPT")/features"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR/simdino"
torchrun --nproc_per_node=${NUM_GPUS} eval_knn.py \
    --arch vit_base --patch_size 16 \
    --pretrained_weights "$CKPT" \
    --checkpoint_key teacher \
    --data_path "$DATA_PATH" \
    --dump_features "$DUMP_DIR" \
    --batch_size_per_gpu 128 \
    --num_workers 10 \
    "$@"
