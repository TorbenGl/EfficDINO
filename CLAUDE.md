# EfficDINO

## Project Overview
Self-supervised learning experiments using SimDINO (v1 codebase in `simdino/`).
Two experiments: (1) SimDINO ViT-B/16 baseline reproduction, (2) EfficEmbedDINO with structured sub-patch tokenization.

## Key Architecture
- **Training**: `simdino/main_dino.py` — entry point, loss classes, training loop
- **Models**: `simdino/vision_transformer.py` — ViT, PatchEmbed, EfficEmbedPatchEmbed, DINOHead
- **Utilities**: `simdino/utils.py` — schedulers, DDP, MultiCropWrapper, checkpoint helpers
- **Evaluation**: `simdino/eval_knn.py` (k-NN), `simdino/eval_linear.py` (linear probe)
- **Scripts**: `scripts/` — SLURM jobs, run_or_resume.sh, eval scripts
- **Plan**: `PLAN.md` — detailed implementation plan with hyperparameters

## Development
- **Local (Windows)**: `pip install -r requirements.txt` — installs CPU-only torch for dev
- **Server (Linux)**: `pip install -r requirements-server.txt` — installs CUDA torch + xformers + cuml

## Training Commands
```bash
# Fresh start — SimDINO baseline
bash scripts/run_or_resume.sh simdino_vitb16_baseline

# Fresh start — EfficEmbedDINO
bash scripts/run_or_resume.sh efficembed_vitb16

# Resume automatically detects latest checkpoint
```

## Key Design Decisions
- Using SimDINO v1 (`simdino/`) not v2 — simpler DDP-based, argparse config
- Gradient accumulation for 4-GPU setup: 64 × 4 × 2 = 512 effective batch
- `--reduce_cov 1` required — all-reduce covariance across GPUs for correct coding rate
- `--local_crops_number 10` — paper uses 10, code default was 8
- `--clip_grad 0.3` — paper uses 0.3, code default was 3.0
- `--lr 0.001` — after linear scaling ×512/256 gives effective lr=0.002

## Dataset

### Option A — Standard ImageFolder (default)
Expects the standard ImageNet folder layout:
```
/path/to/imagenet/
  train/  n01440764/  ...
  val/    n01440764/  ...
```
Set `DATA_PATH=/path/to/imagenet/train` and `EVAL_DATA_PATH=/path/to/imagenet`.

### Option B — HuggingFace ImageNet-1k (v2 scripts)

**Download first:**
```bash
HF_TOKEN=<token> python scripts/download_imagenet.py --save_path /datasets/imagenet-1k
# or via env:
HF_TOKEN=<token> IMAGENET_SAVE_PATH=/datasets/imagenet-1k python scripts/download_imagenet.py
```

**Train using HF dataset (v2 scripts):**
```bash
HF_DATA_PATH=/datasets/imagenet-1k bash scripts/run_or_resume_v2.sh simdino_vitb16_baseline
HF_DATA_PATH=/datasets/imagenet-1k bash scripts/run_or_resume_v2.sh efficembed_vitb16
```

The v2 scripts pass `--dataset_type hf_imagenet` and point `--data_path` / `--eval_data_path`
at the same HF snapshot root (train + validation splits loaded automatically).

See `PLAN.md` section 12 for the full v2 implementation plan.

## WandB
- Set `WANDB_API_KEY` env var before training
- Project: `efficdino`
- Logs: loss, comp_loss, expa_loss, lr, wd, grad_norm, ema_momentum, epoch_time, knn_top1

## Checkpointing
- `checkpoint.pth` — latest (overwritten each epoch)
- `checkpoint_NNNN.pth` — periodic (every `--saveckp_freq` epochs, keeps last 3)
- `checkpoint_best.pth` — best k-NN accuracy
- `checkpoint_interrupted.pth` — on SIGTERM/SIGUSR1
- Full state: student, teacher, optimizer, scaler, loss, RNG states, best metrics
