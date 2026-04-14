#!/bin/bash
# =============================================================================
# SLURM job: EfficEmbedDINOv2 ViT-B/16 — DINOv2 pipeline, HF ImageNet
#
# Runs simdinov2/train/train.py with EfficEmbedPatchEmbed tokenization
# (MCR loss, auto-resume from output-dir).
#
# BEFORE SUBMITTING:
#   1. Set HF_DATA_PATH to your imagenet-1k snapshot directory
#   2. Set WANDB_API_KEY (or export it in your environment)
#   3. Activate your venv or set VENV_PATH
# =============================================================================
#SBATCH --job-name=efficembed-vitb16
#SBATCH --partition=kisski
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/efficembed-v2-%j.out
#SBATCH --error=logs/efficembed-v2-%j.err
#SBATCH --signal=B:TERM@120

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
HF_DATA_PATH="${HF_DATA_PATH:?HF_DATA_PATH must be set to the imagenet-1k snapshot directory}"

mkdir -p logs

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export MASTER_PORT="${MASTER_PORT:-29501}"

echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Num GPUs:        $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "HF_DATA_PATH:    $HF_DATA_PATH"

cd "$REPO_DIR"
exec bash scripts/run_or_resume_dinov2.sh efficembed_vitb16
