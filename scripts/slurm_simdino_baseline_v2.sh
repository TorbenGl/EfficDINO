#!/bin/bash
# =============================================================================
# SLURM job: SimDINO ViT-B/16 baseline — HuggingFace ImageNet-1k (v2)
#
# Uses HF snapshot_download dataset instead of ImageFolder layout.
#
# BEFORE SUBMITTING:
#   1. Set HF_DATA_PATH to your imagenet-1k snapshot directory
#   2. Set partition/nodelist for your cluster
#   3. Activate your venv or set VENV_PATH
#   4. Download data first: python scripts/download_imagenet.py --save_path <HF_DATA_PATH>
# =============================================================================
#SBATCH --job-name=simdino-vitb16-baseline-v2
#SBATCH --partition=gpu-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/simdino-baseline-v2-%j.out
#SBATCH --error=logs/simdino-baseline-v2-%j.err
#SBATCH --signal=B:TERM@120

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV_DIR="${VENV_PATH:-$REPO_DIR/.venv}"
HF_DATA_PATH="${HF_DATA_PATH:?HF_DATA_PATH must be set to the imagenet-1k snapshot directory}"

mkdir -p logs

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo "Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Num GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "HF_DATA_PATH: $HF_DATA_PATH"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export MASTER_PORT="${MASTER_PORT:-29500}"

cd "$REPO_DIR"
exec bash scripts/run_or_resume_v2.sh simdino_vitb16_baseline
