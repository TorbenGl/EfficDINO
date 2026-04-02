# EfficDINO — Quickstart

## Files Created / Modified

### New files

| File | Purpose |
|---|---|
| `PLAN.md` | Detailed implementation plan with hyperparameters, commands, and checklist |
| `CLAUDE.md` | Project guide for quick reference |
| `requirements-server.txt` | Linux GPU deps (CUDA 11.8 + xformers + cuml) |
| `scripts/run_or_resume.sh` | Auto-detect & resume or start fresh |
| `scripts/slurm_simdino_baseline.sh` | SLURM job for Exp 1 |
| `scripts/slurm_efficembed.sh` | SLURM job for Exp 2 |
| `scripts/eval_knn.sh` | k-NN evaluation wrapper |
| `scripts/eval_linear.sh` | Linear probe wrapper |

### Modified files

| File | Changes |
|---|---|
| `simdino/vision_transformer.py` | Added `EfficEmbedPatchEmbed` class (Conv k=4 + GELU + space-to-depth); `VisionTransformer` accepts `embed_type`; factory funcs pass through |
| `simdino/main_dino.py` | Gradient accumulation (`--grad_accum_steps` + `no_sync`), SIGTERM handler, checkpoint with RNG states + rotation + best tracking, WandB resume with `wandb_run_id`, periodic k-NN eval, new CLI args |
| `simdino/eval_knn.py` | Added `--embed_type`, `--sub_patch_size`, `--sub_patch_channels` |
| `simdino/eval_linear.py` | Same EfficEmbed args |
| `requirements.txt` | Stripped to CPU-only base (Windows-compatible) |
| `.gitignore` | Added `checkpoints/`, `logs/`, `features/` |

## Key things to configure before running

1. **Set `DATA_PATH`** env var to your ImageNet directory (with `train/` and `val/` subdirs)
2. **Set `WANDB_API_KEY`** env var for WandB logging
3. **Edit SLURM `--partition`** in the SLURM scripts to match your cluster
4. **Install on server**: `pip install -r requirements-server.txt`
5. **Install locally**: `pip install -r requirements.txt`

## Quick start

```bash
# On the server:
export DATA_PATH=/path/to/imagenet/train
export EVAL_DATA_PATH=/path/to/imagenet
export WANDB_API_KEY=your_key

# Exp 1 — SimDINO baseline:
sbatch scripts/slurm_simdino_baseline.sh

# Exp 2 — EfficEmbedDINO:
sbatch scripts/slurm_efficembed.sh

# Or manually without SLURM:
bash scripts/run_or_resume.sh simdino_vitb16_baseline
bash scripts/run_or_resume.sh efficembed_vitb16
```
