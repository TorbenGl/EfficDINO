# EfficDINO — Implementation Plan

## Overview

Two experiments on ImageNet-1K for 100 epochs using the SimDINO v1 codebase (`simdino/`):

| Experiment | Description | Target |
|---|---|---|
| **Exp 1 — SimDINO baseline** | Reproduce ViT-B/16 from paper | 74.9 k-NN, 77.3 linear probe |
| **Exp 2 — EfficEmbedDINO** | Structured sub-patch tokenization | Compare with Exp 1 |

Hardware: **4× A100/H100 (80 GB)** — paper uses 8 GPUs, so we use gradient accumulation.

---

## 1. Repo Analysis Summary

### Codebase Structure (using `simdino/` v1 — simpler DDP-based approach)

| Component | File | Key Lines |
|---|---|---|
| Training entry | `simdino/main_dino.py` | `train_dino()` L152, `train_one_epoch()` L361 |
| ViT backbone | `simdino/vision_transformer.py` | `VisionTransformer` L134, `PatchEmbed` L116 |
| Projector head | `simdino/vision_transformer.py` | `DINOHead` L258 |
| Loss (MCR) | `simdino/main_dino.py` | `MCRLoss` L497, compression L521, expansion L536 |
| Utilities | `simdino/utils.py` | Schedulers, DDP init, MultiCropWrapper |
| k-NN eval | `simdino/eval_knn.py` | Feature extraction + k-NN classifier |
| Linear eval | `simdino/eval_linear.py` | Linear probe on frozen features |

### Existing Checkpoint/Resume — GAPS

**Already saved:** student, teacher, optimizer, epoch, args, dino_loss, fp16_scaler
**Missing:**
- [x] RNG states (torch, numpy, python, CUDA) → Added
- [x] Best metrics tracking → Added
- [x] SIGTERM/SIGUSR1 handler → Added
- [x] Checkpoint rotation (keep last 3 + best) → Added
- [x] Gradient accumulation state → Added (via proper epoch/iter tracking)

### Batch Size & Gradient Accumulation — GAP

- Code uses `--batch_size_per_gpu` (per-GPU). No gradient accumulation.
- Paper: 64 × 8 GPUs = 512 effective.
- Our setup: 64 × 4 GPUs × 2 accum steps = 512 effective. → Added `--grad_accum_steps`.
- LR scaling formula: `lr * (batch_size_per_gpu * world_size * grad_accum_steps) / 256`

### Covariance in Coding Rate — CRITICAL

- `reduce_cov` flag exists (default=0, local only).
- **Must set `--reduce_cov 1`** to all-reduce covariance across GPUs for correct behavior.
- With gradient accumulation, covariance is computed per micro-batch (not accumulated), which is acceptable since the coding rate sees `batch_size_per_gpu` samples from each GPU, and all-reduce aggregates across GPUs.

---

## 2. SimDINO Baseline — Exact Hyperparameters

From Table 4 (page 15) and pages 4-6 of the paper:

```
Architecture:
  --arch vit_base                    # ViT-B/16
  --patch_size 16
  --z_dim 256                        # bottleneck/output dim
  --hidden_dim 2048                  # projector hidden dim
  --out_dim 65536                    # (unused, DINO head removed)
  --use_simdino True                 # Use MCR loss, not DINO CE loss
  --drop_path_rate 0.1

Loss:
  --eps 0.5                          # ε for coding rate
  --coeff 1.0                        # γ coefficient for compression
  --reduce_cov 1                     # all-reduce covariance across GPUs
  --expa_type 1                      # smoothing: (student+teacher)/2

Optimization:
  --optimizer adamw
  --lr 0.001                         # → effective 0.002 after scaling (×512/256)
  --weight_decay 0.04
  --weight_decay_end 0.4
  --warmup_epochs 10
  --clip_grad 0.3                    # paper says 0.3 (code default is 3.0!)
  --batch_size_per_gpu 64
  --grad_accum_steps 2               # 64×4×2 = 512 effective
  --epochs 100
  --use_fp16 True

Teacher EMA:
  --momentum_teacher 0.996           # cosine schedule → 1.0

Data augmentation:
  --global_crops_scale 0.4 1.0
  --local_crops_number 10            # paper uses 10 (code default is 8!)
  --local_crops_scale 0.05 0.4
```

### LR Calculation Verification

```
lr_base = 0.001 (--lr argument)
effective_batch = 64 * 4 * 2 = 512
lr_actual = 0.001 * (512 / 256) = 0.002 ✓ (matches paper Table 4)
```

---

## 3. Distributed Training — 4 GPU Setup

### Launch Command
```bash
torchrun --nproc_per_node=4 simdino/main_dino.py [args]
```

### Gradient Accumulation Implementation

Added to `train_one_epoch()`:
- Forward/backward on micro-batches without optimizer step
- Use DDP `no_sync()` context for accumulation steps (skip all-reduce on non-final steps)
- Only step optimizer + EMA update on final accumulation step
- LR/WD schedules indexed by global step (not micro-step)
- Gradient clipping applied once after accumulation

### Batch Size Options

| Option | per_gpu | GPUs | accum | effective | Memory estimate |
|---|---|---|---|---|---|
| A (safe) | 64 | 4 | 2 | 512 | ~45 GB/GPU |
| B (aggressive) | 128 | 4 | 1 | 512 | ~70 GB/GPU |
| C (fallback) | 64 | 4 | 1 | 256 | ~45 GB/GPU |

With 12 views (2 global 224² + 10 local 96²) for ViT-B, Option A is recommended.

---

## 4. Checkpoint/Resume — Full Implementation

### Checkpoint Dict Structure

```python
checkpoint = {
    'epoch': epoch + 1,
    'global_step': global_step,
    'student': student.state_dict(),
    'teacher': teacher.state_dict(),
    'optimizer': optimizer.state_dict(),
    'dino_loss': dino_loss.state_dict(),
    'fp16_scaler': fp16_scaler.state_dict() if fp16_scaler else None,
    'args': args,
    'best_knn': best_knn,
    'rng_states': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(),
    },
}
```

### Save Points
- End of every epoch: `checkpoint.pth` (overwritten)
- Every `--saveckp_freq` epochs: `checkpoint_NNNN.pth` (kept with rotation)
- Best k-NN accuracy: `checkpoint_best.pth`
- On SIGTERM/SIGUSR1: `checkpoint_interrupted.pth`

### Checkpoint Rotation
- Keep last 3 periodic checkpoints + best + latest
- Auto-delete older periodic checkpoints

### SIGTERM Handler
- Register signal handlers for SIGTERM and SIGUSR1
- On signal: save checkpoint, then exit gracefully
- Implemented via global flag checked at end of each epoch

---

## 5. EfficEmbedDINO — Patch Embedding Design

### Architecture

```
Standard ViT-B/16 PatchEmbed:
  Conv2d(3, 768, kernel_size=16, stride=16)
  (B, 3, 224, 224) → (B, 768, 14, 14) → (B, 196, 768)

EfficEmbedPatchEmbed (k=4, C=48):
  Step 1: Conv2d(3, 48, kernel_size=4, stride=4) + GELU
    (B, 3, 224, 224) → (B, 48, 56, 56)

  Step 2: Space-to-depth (parameter-free reshape)
    (B, 48, 56, 56)
    → (B, 48, 14, 4, 14, 4)    # split spatial dims by fold_factor=16/4=4
    → (B, 48*16, 14, 14)        # merge sub-patch positions into channels
    = (B, 768, 14, 14)

  Result: (B, 768, 14, 14) → flatten → (B, 196, 768)

For local views (96×96):
  Conv: (B, 3, 96, 96) → (B, 48, 24, 24)
  Space-to-depth: (B, 48, 24, 24) → (B, 48, 6, 4, 6, 4) → (B, 768, 6, 6)
  Result: (B, 768, 6, 6) → flatten → (B, 36, 768)
```

### Key Properties
- **C=48, k=4**: D = 48 × (16/4)² = 48 × 16 = 768 = ViT-B embed_dim. No projection needed.
- **Token count unchanged**: 14×14=196 for 224², 6×6=36 for 96². Positional embeddings work as-is.
- **GELU activation** after small conv (one nonlinearity for sub-patch feature learning).
- **No BatchNorm/LayerNorm** after conv (keep it simple, one variable at a time).
- **All other hyperparameters identical** to Experiment 1.

### Implementation in `vision_transformer.py`

New class `EfficEmbedPatchEmbed(nn.Module)`:
- `__init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, sub_patch_size=4)`
- Validates: `patch_size % sub_patch_size == 0`
- Validates: `in_chans_sub * fold_factor² == embed_dim` (or adds 1×1 projection if needed)
- `forward(self, x)`: conv → GELU → space-to-depth → flatten → transpose

### Integration Points
- `VisionTransformer.__init__`: Accept `--embed_type` flag (`standard` or `efficembed`)
- `--sub_patch_size` argument (default 4)
- `--sub_patch_channels` argument (default 48 for ViT-B, auto-computed if not specified)
- Both student and teacher use the same embedding type
- Eval scripts: pass `--embed_type efficembed --sub_patch_size 4` when evaluating EfficEmbed checkpoints

---

## 6. Training Schedule

### Chunked Runs (resume between chunks)

```
Exp 1 — SimDINO baseline:
  Chunk A: epochs 0-29   (~2-3 days on 4×A100)
  Chunk B: epochs 30-59  (resume from checkpoint)
  Chunk C: epochs 60-99  (resume from checkpoint)
  → Evaluate: k-NN + linear probe on best checkpoint

Exp 2 — EfficEmbedDINO:
  Chunk A: epochs 0-29
  Chunk B: epochs 30-59
  Chunk C: epochs 60-99
  → Evaluate: k-NN + linear probe
  → Compare with Exp 1
```

---

## 7. Monitoring & Sanity Checks

### WandB Logging (per iteration)
- `loss`, `comp_loss`, `expa_loss` — total, compression, expansion
- `lr`, `wd` — current learning rate and weight decay
- `grad_norm` — gradient norm before clipping
- `ema_momentum` — current teacher EMA momentum value
- `epoch_time` — wall-clock seconds per epoch
- `gpu_memory_mb` — peak GPU memory

### Periodic k-NN Eval
- Every `--eval_freq` epochs (default 10), run k-NN on val set
- Log `knn_top1`, `knn_top5` to WandB
- Save `checkpoint_best.pth` if k-NN improves

### Resume Sanity Check
- After resume, log first-iteration loss
- Compare with last-iteration loss from previous chunk (logged in `log.txt`)
- If difference > 10%, print warning

---

## 8. Pre-Launch Checklist

- [ ] Verify token shapes: 196 for 224×224, 36 for 96×96 — BOTH standard and EfficEmbed
- [ ] Verify loss matches Algorithm 1 (page 13)
- [ ] Verify coding rate formula uses `p/(m*N*eps)` scaling (line 551)
- [ ] Verify teacher EMA includes all parameters
- [ ] Verify checkpoint saves/loads correctly (save epoch 1, kill, resume, verify epoch 2)
- [ ] Verify gradient accumulation: effective batch = 512
- [ ] Verify covariance all-reduced with `--reduce_cov 1`
- [ ] Run 1 epoch of both experiments, check loss values are reasonable
- [ ] Verify space-to-depth works for BOTH global (224²) and local (96²)
- [ ] Verify SIGTERM handler triggers checkpoint save
- [ ] Verify WandB logging appends on resume (same run ID)

---

## 9. File Changes Summary

| File | Change | Status |
|---|---|---|
| `simdino/vision_transformer.py` | Added `EfficEmbedPatchEmbed` class (L134-212); modified `VisionTransformer.__init__` to accept `embed_type` (L220); updated factory funcs `vit_tiny/small/base` (L320+) | ✅ Done |
| `simdino/main_dino.py` | Added: gradient accumulation (`--grad_accum_steps`, `no_sync`), SIGTERM handler, `_build_checkpoint` with RNG states, `_rotate_checkpoints`, `_run_knn_eval`, WandB resume with `wandb_run_id`, new CLI args (`--embed_type`, `--eval_freq`, `--wandb_*`, etc.) | ✅ Done |
| `simdino/utils.py` | No changes needed — checkpoint rotation + RNG helpers implemented in `main_dino.py` directly | ✅ N/A |
| `simdino/eval_knn.py` | Added `--embed_type`, `--sub_patch_size`, `--sub_patch_channels` args; updated model construction | ✅ Done |
| `simdino/eval_linear.py` | Added `--embed_type`, `--sub_patch_size`, `--sub_patch_channels` args; updated model construction | ✅ Done |
| `requirements.txt` | Base requirements (Windows-compatible, no CUDA deps) | ✅ Done |
| `requirements-server.txt` | Linux GPU server requirements (CUDA 11.8 + xformers + cuml) | ✅ Done |
| `scripts/run_or_resume.sh` | Convenience wrapper: auto-detects experiment type, resumes or starts fresh | ✅ Done |
| `scripts/slurm_simdino_baseline.sh` | SLURM job for Exp 1 (4×GPU, 72h, SIGTERM@120s) | ✅ Done |
| `scripts/slurm_efficembed.sh` | SLURM job for Exp 2 (4×GPU, 72h, SIGTERM@120s) | ✅ Done |
| `scripts/eval_knn.sh` | k-NN evaluation wrapper | ✅ Done |
| `scripts/eval_linear.sh` | Linear probe evaluation wrapper | ✅ Done |
| `CLAUDE.md` | Project guide | ✅ Done |
| `.gitignore` | Added `checkpoints/`, `logs/`, `features/` | ✅ Done |

---

## 10. Training Commands

### Experiment 1 — SimDINO Baseline (fresh start)
```bash
torchrun --nproc_per_node=4 simdino/main_dino.py \
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
    --data_path /path/to/imagenet/train \
    --output_dir ./checkpoints/simdino_vitb16_baseline \
    --saveckp_freq 10 --eval_freq 10 \
    --eval_data_path /path/to/imagenet \
    --track_wandb --wandb_project efficdino --wandb_run_name simdino_vitb16_baseline
```

### Experiment 2 — EfficEmbedDINO (fresh start)
```bash
torchrun --nproc_per_node=4 simdino/main_dino.py \
    --arch vit_base --patch_size 16 \
    --embed_type efficembed --sub_patch_size 4 \
    --epochs 100 --batch_size_per_gpu 64 --grad_accum_steps 2 \
    --lr 0.001 --weight_decay 0.04 --weight_decay_end 0.4 \
    --warmup_epochs 10 --clip_grad 0.3 \
    --momentum_teacher 0.996 \
    --local_crops_number 10 \
    --global_crops_scale 0.4 1.0 --local_crops_scale 0.05 0.4 \
    --use_simdino True --eps 0.5 --coeff 1.0 --reduce_cov 1 --expa_type 1 \
    --use_fp16 True --compile True \
    --z_dim 256 --hidden_dim 2048 \
    --data_path /path/to/imagenet/train \
    --output_dir ./checkpoints/efficembed_vitb16 \
    --saveckp_freq 10 --eval_freq 10 \
    --eval_data_path /path/to/imagenet \
    --track_wandb --wandb_project efficdino --wandb_run_name efficembed_vitb16
```

### Resume (either experiment)
```bash
# Auto-detect latest checkpoint:
bash scripts/run_or_resume.sh simdino_vitb16_baseline
bash scripts/run_or_resume.sh efficembed_vitb16
```

### Evaluation Only
```bash
# k-NN
torchrun --nproc_per_node=4 simdino/eval_knn.py \
    --arch vit_base --patch_size 16 \
    --pretrained_weights ./checkpoints/simdino_vitb16_baseline/checkpoint_best.pth \
    --checkpoint_key teacher \
    --data_path /path/to/imagenet \
    --dump_features ./checkpoints/simdino_vitb16_baseline/features

# For EfficEmbed:
torchrun --nproc_per_node=4 simdino/eval_knn.py \
    --arch vit_base --patch_size 16 \
    --embed_type efficembed --sub_patch_size 4 \
    --pretrained_weights ./checkpoints/efficembed_vitb16/checkpoint_best.pth \
    --checkpoint_key teacher \
    --data_path /path/to/imagenet \
    --dump_features ./checkpoints/efficembed_vitb16/features

# Linear probe
torchrun --nproc_per_node=4 simdino/eval_linear.py \
    --arch vit_base --patch_size 16 \
    --pretrained_weights ./checkpoints/simdino_vitb16_baseline/checkpoint_best.pth \
    --checkpoint_key teacher \
    --data_path /path/to/imagenet \
    --output_dir ./checkpoints/simdino_vitb16_baseline/linear_eval \
    --epochs 100 --lr 0.001 --batch_size_per_gpu 256 \
    --n_last_blocks 1 --avgpool_patchtokens True
```

---

## 12. v2 — HuggingFace ImageNet-1k Support

### Goal

Allow training and evaluation directly from an HF-downloaded `ILSVRC/imagenet-1k` dataset
stored at any local path, without requiring the standard `train/` / `val/` ImageFolder layout.
All existing behaviour (ImageFolder) is preserved as the default.

### Download

```bash
python scripts/download_imagenet.py --save_path /datasets/imagenet-1k
# or
HF_TOKEN=<token> IMAGENET_SAVE_PATH=/datasets/imagenet-1k python scripts/download_imagenet.py
```

The script uses `huggingface_hub.snapshot_download` with `resume_download=True` so interrupted
downloads continue from where they left off.

---

### Code Changes

#### New CLI flag — all three training/eval scripts

```
--dataset_type  {imagefolder, hf_imagenet}   (default: imagefolder)
--data_path                                  for hf_imagenet: root of snapshot_download output
--eval_data_path                             for hf_imagenet: same root (train + validation splits)
```

#### `simdino/main_dino.py`

1. Add `HFImageNetDataset(torch.utils.data.Dataset)` wrapper:
   - `__init__(self, hf_dataset, transform=None)` — wraps a HF `datasets.Dataset`
   - `__getitem__`: returns `(transform(sample['image'].convert('RGB')), sample['label'])`
   - `targets` property: `[s['label'] for s in self.dataset]` — used for label extraction
2. Add `_load_hf_dataset(data_path, split)`:
   ```python
   from datasets import load_dataset
   return load_dataset(data_path, split=split, trust_remote_code=False)
   ```
3. In `train_dino()`, gate dataset construction on `args.dataset_type`:
   ```python
   if args.dataset_type == 'hf_imagenet':
       hf_ds = _load_hf_dataset(args.data_path, 'train')
       dataset = HFImageNetDataset(hf_ds, transform=transform)
   else:
       dataset = datasets.ImageFolder(args.data_path, transform=transform)
   ```
4. In `_run_knn_eval()`, same gate for train + validation splits:
   - HF split name is `"validation"` (not `"val"`)
   - Labels via `dataset.targets` (same property for both ImageFolder and HFImageNetDataset)

#### `simdino/eval_knn.py`

1. Same `--dataset_type` flag.
2. Same `HFImageNetDataset` / `_load_hf_dataset` helpers (share via import or duplicate).
3. `ReturnIndexDataset` wrapper:
   - For `imagefolder`: existing subclass of `datasets.ImageFolder`
   - For `hf_imagenet`: new `ReturnIndexHFDataset` that returns `(img, idx)` and has `.targets`
4. Label extraction at L91-92:
   ```python
   # current (imagefolder):
   train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
   # new (hf_imagenet):
   train_labels = torch.tensor(dataset_train.targets).long()
   ```
   Use `.targets` in both cases — add `targets` property to `ReturnIndexDataset` too.

#### `simdino/eval_linear.py`

1. Same `--dataset_type` flag.
2. Replace `datasets.ImageFolder(os.path.join(args.data_path, split))` with:
   ```python
   if args.dataset_type == 'hf_imagenet':
       hf_ds = _load_hf_dataset(args.data_path, 'train' if split == 'train' else 'validation')
       dataset = HFImageNetDataset(hf_ds, transform=transform)
   else:
       dataset = datasets.ImageFolder(os.path.join(args.data_path, split), transform=transform)
   ```

---

### New / Updated Scripts

| File | Change |
|---|---|
| `scripts/download_imagenet.py` | ✅ New — resumable downloader for ILSVRC/imagenet-1k |
| `scripts/run_or_resume_v2.sh` | New — same as `run_or_resume.sh` but sets `--dataset_type hf_imagenet` and uses `HF_DATA_PATH` |
| `scripts/slurm_simdino_baseline_v2.sh` | New — SLURM job for Exp 1, HF dataset path |
| `scripts/slurm_efficembed_v2.sh` | New — SLURM job for Exp 2, HF dataset path |
| `scripts/eval_knn_v2.sh` | New — k-NN eval wrapper using HF dataset |
| `scripts/eval_linear_v2.sh` | New — linear probe eval wrapper using HF dataset |

#### `scripts/run_or_resume_v2.sh` key differences from v1

```bash
HF_DATA_PATH="${HF_DATA_PATH:-/datasets/imagenet-1k}"
# ...
DATASET_ARGS="--dataset_type hf_imagenet --data_path ${HF_DATA_PATH} --eval_data_path ${HF_DATA_PATH}"
# replaces:
#   --data_path ${DATA_PATH} --eval_data_path ${EVAL_DATA_PATH}
```

#### SLURM v2 scripts key differences

- `DATA_PATH` replaced by `HF_DATA_PATH`
- No separate `EVAL_DATA_PATH` (same root used for both splits)
- Call `run_or_resume_v2.sh` instead of `run_or_resume.sh`

---

### Requirements

Add to `requirements-server.txt`:
```
datasets>=2.18.0
```

Add to `requirements.txt` (CPU/dev):
```
datasets>=2.18.0
```

---

### Status

| Item | Status |
|---|---|
| `scripts/download_imagenet.py` | ✅ Done |
| `--dataset_type` flag in `main_dino.py` | ✅ Done |
| `HFImageNetDataset` + `_load_hf_dataset` in `main_dino.py` | ✅ Done |
| `--dataset_type` + HF loading in `eval_knn.py` | ✅ Done |
| `--dataset_type` + HF loading in `eval_linear.py` | ✅ Done |
| `scripts/run_or_resume_v2.sh` | ✅ Done |
| `scripts/slurm_simdino_baseline_v2.sh` | ✅ Done |
| `scripts/slurm_efficembed_v2.sh` | ✅ Done |
| `scripts/eval_knn_v2.sh` | ✅ Done |
| `scripts/eval_linear_v2.sh` | ✅ Done |
| `requirements*.txt` — add `datasets` + `huggingface_hub` | ✅ Done |

---

## 11. GPU Memory Estimates

ViT-B/16 with 12 views (2×224² + 10×96²), fp16, batch_size_per_gpu=64:

| Component | Memory |
|---|---|
| Student model params | ~0.5 GB |
| Teacher model params (no grad) | ~0.5 GB |
| Optimizer states (AdamW) | ~1.5 GB |
| Student activations (12 views) | ~25-35 GB |
| Teacher activations (2 views) | ~3-5 GB |
| Gradients | ~0.5 GB |
| **Total estimate** | **~35-45 GB** |

Fits comfortably in 80 GB A100/H100 with batch_size=64. Option B (batch=128) at ~70 GB is tight.

### Time-per-epoch estimate
- ImageNet-1K: ~1.28M images
- Batch 512 effective: ~2500 iterations per epoch
- ViT-B forward+backward: ~0.3-0.5s per iteration on 4×A100
- **~15-20 minutes per epoch** → ~25-33 hours total for 100 epochs
- Each 30-epoch chunk: ~8-10 hours
