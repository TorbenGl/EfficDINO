#!/bin/bash
# =============================================================================
# smoke_test.sh — Quick smoke test for all 4 training modes (20 iterations each)
#
# Intended to run inside an interactive KISSKI session to catch import errors,
# config issues, and data loading problems before a full job submission.
#
# Get an interactive session first:
#   srun --partition=kisski --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 \
#        --cpus-per-task=10 --mem=64G --time=01:00:00 --pty bash
#
# Then run:
#   cd ~/EfficDINO && source .venv/bin/activate
#   HF_DATA_PATH=/path/to/hf-imagenet WANDB_API_KEY=<key> bash scripts/smoke_test.sh
#
# Modes tested:
#   1. simdino baseline     (simdino/main_dino.py, HF data, v1 pipeline)
#   2. simdino efficembed   (simdino/main_dino.py, HF data + EfficEmbedPatchEmbed)
#   3. simdinov2 baseline   (simdinov2/train/train.py, HF data, MCR loss)
#   4. simdinov2 efficembed (simdinov2/train/train.py, HF data + EfficEmbedPatchEmbed)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HF_DATA_PATH="${HF_DATA_PATH:?HF_DATA_PATH must be set}"
NUM_GPUS="${NUM_GPUS:-4}"
SMOKE_DIR="/tmp/efficdino_smoke_$$"

PASS=0
FAIL=0

run_mode() {
    local name="$1"
    local outdir="${SMOKE_DIR}/${name}"
    local logfile="${SMOKE_DIR}/${name}.log"
    mkdir -p "$outdir"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: ${name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    shift
    if torchrun --nproc_per_node="${NUM_GPUS}" "$@" >"$logfile" 2>&1; then
        echo "  ✓ PASS: ${name}"
        PASS=$((PASS + 1))
    else
        echo "  ✗ FAIL: ${name}"
        echo "  --- last 30 lines ---"
        tail -30 "$logfile"
        FAIL=$((FAIL + 1))
    fi
}

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Mode 1: simdino baseline (v1 pipeline, HF data)
# ---------------------------------------------------------------------------
run_mode "simdino_baseline" \
    simdino/main_dino.py \
    --arch vit_base --patch_size 16 \
    --epochs 1 --batch_size_per_gpu 4 --grad_accum_steps 1 \
    --lr 0.001 --weight_decay 0.04 --weight_decay_end 0.4 \
    --warmup_epochs 0 --clip_grad 0.3 \
    --momentum_teacher 0.996 \
    --local_crops_number 2 \
    --global_crops_scale 0.4 1.0 --local_crops_scale 0.05 0.4 \
    --use_simdino True --eps 0.5 --coeff 1.0 --reduce_cov 1 --expa_type 1 \
    --use_fp16 True \
    --z_dim 256 --hidden_dim 2048 \
    --dataset_type hf_imagenet \
    --data_path "${HF_DATA_PATH}" \
    --eval_data_path "${HF_DATA_PATH}" \
    --output_dir "${SMOKE_DIR}/simdino_baseline" \
    --saveckp_freq 999 --eval_freq 999

# ---------------------------------------------------------------------------
# Mode 2: simdino efficembed (v1 pipeline, HF data + EfficEmbed)
# ---------------------------------------------------------------------------
run_mode "simdino_efficembed" \
    simdino/main_dino.py \
    --arch vit_base --patch_size 16 \
    --embed_type efficembed --sub_patch_size 4 \
    --epochs 1 --batch_size_per_gpu 4 --grad_accum_steps 1 \
    --lr 0.001 --weight_decay 0.04 --weight_decay_end 0.4 \
    --warmup_epochs 0 --clip_grad 0.3 \
    --momentum_teacher 0.996 \
    --local_crops_number 2 \
    --global_crops_scale 0.4 1.0 --local_crops_scale 0.05 0.4 \
    --use_simdino True --eps 0.5 --coeff 1.0 --reduce_cov 1 --expa_type 1 \
    --use_fp16 True \
    --z_dim 256 --hidden_dim 2048 \
    --dataset_type hf_imagenet \
    --data_path "${HF_DATA_PATH}" \
    --eval_data_path "${HF_DATA_PATH}" \
    --output_dir "${SMOKE_DIR}/simdino_efficembed" \
    --saveckp_freq 999 --eval_freq 999

# ---------------------------------------------------------------------------
# Mode 3: simdinov2 baseline (DINOv2 pipeline, HF data)
# ---------------------------------------------------------------------------
run_mode "simdinov2_baseline" \
    simdinov2/train/train.py \
    --base-config simdino_config \
    --config-file "${REPO_DIR}/simdinov2/configs/train/vitb16_simdino.yaml" \
    --output-dir "${SMOKE_DIR}/simdinov2_baseline" \
    --no-resume \
    "train.dataset_path=HFImageNet:split=TRAIN:root=${HF_DATA_PATH}" \
    "train.batch_size_per_gpu=4" \
    "train.OFFICIAL_EPOCH_LENGTH=20" \
    "optim.epochs=1" \
    "optim.warmup_epochs=0"

# ---------------------------------------------------------------------------
# Mode 4: simdinov2 efficembed (DINOv2 pipeline, HF data + EfficEmbed)
# ---------------------------------------------------------------------------
run_mode "simdinov2_efficembed" \
    simdinov2/train/train.py \
    --base-config simdino_config \
    --config-file "${REPO_DIR}/simdinov2/configs/train/vitb16_efficembed.yaml" \
    --output-dir "${SMOKE_DIR}/simdinov2_efficembed" \
    --no-resume \
    "train.dataset_path=HFImageNet:split=TRAIN:root=${HF_DATA_PATH}" \
    "train.batch_size_per_gpu=4" \
    "train.OFFICIAL_EPOCH_LENGTH=20" \
    "optim.epochs=1" \
    "optim.warmup_epochs=0"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Smoke test results: ${PASS} passed, ${FAIL} failed"
echo "  Logs: ${SMOKE_DIR}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
