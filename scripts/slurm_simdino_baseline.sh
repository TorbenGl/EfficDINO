#!/bin/bash
# =============================================================================
# SLURM job: SimDINO ViT-B/16 baseline reproduction (Experiment 1)
#
# 4× A100/H100 80GB, 100 epochs, effective batch size 512
# Supports automatic resume — just re-submit the same script.
#
# BEFORE SUBMITTING:
#   1. Set DATA_PATH to your ImageNet directory
#   2. Set partition/nodelist for your cluster
#   3. Activate your venv or set VENV_PATH
# =============================================================================
#SBATCH --job-name=simdino-vitb16-baseline
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/simdino-baseline-%j.out
#SBATCH --error=logs/simdino-baseline-%j.err
#SBATCH --signal=B:TERM@120

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
DATA_PATH="${DATA_PATH:-/datasets/imagenet/train}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/datasets/imagenet}"

mkdir -p logs

# Activate environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Num GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export MASTER_PORT="${MASTER_PORT:-29500}"

cd "$REPO_DIR"
exec bash scripts/run_or_resume.sh simdino_vitb16_baseline
