# EfficDINO — Run Process

From zero to running training jobs on KISSKI HPC.

---

## Step 1 — Sync code to server

**Local machine:**
```bash
git add -A && git commit -m "your message"
git push
```

**Server (`glogin9`):**
```bash
cd ~/EfficDINO
git pull
```

---

## Step 2 — Create venv and install dependencies

```bash
cd ~/EfficDINO
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-server.txt --index-strategy unsafe-best-match
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Step 3 — Set persistent environment variables

Add to `~/.bashrc` (once only):
```bash
echo 'export WANDB_API_KEY="your_wandb_key"'   >> ~/.bashrc
echo 'export HF_TOKEN="your_hf_token"'          >> ~/.bashrc
source ~/.bashrc
```

Get your keys at:
- WandB: https://wandb.ai/authorize
- HuggingFace: https://huggingface.co/settings/tokens (needs read access to `ILSVRC/imagenet-1k`)

---

## Step 4 — Download ImageNet-1k (~150 GB)

Run on the login node (resumable if interrupted):
```bash
cd ~/EfficDINO
HF_TOKEN=$HF_TOKEN python scripts/download_imagenet.py --save_path /path/to/data/imagenet-1k
```

Once complete, add to `~/.bashrc`:
```bash
echo 'export HF_DATA_PATH="/path/to/data/imagenet-1k"' >> ~/.bashrc
source ~/.bashrc
```

> **Important:** use the full absolute path — do not use `~` or `$HOME`. The `datasets` library
> does not expand tildes and will throw `HFValidationError` if the path contains `~`.
> 
> Bad:  `export HF_DATA_PATH="~/imagenet-1k"`
> Good: `export HF_DATA_PATH="/trinity/home/tglobisch/imagenet-1k"`

---

## Step 5 — Smoke test (interactive session)

Allocate 4 GPUs interactively (`--qos=2h` gets you to the front of the queue):
```bash
srun --partition=kisski --qos=2h --nodes=1 --ntasks-per-node=4 \
     --gres=gpu:A100:4 --cpus-per-task=16 --mem=64G --time=02:00:00 --pty bash
```

Inside the session:
```bash
cd ~/EfficDINO
source .venv/bin/activate
bash scripts/smoke_test.sh
```

All 4 modes should print `✓ PASS`:
- simdino baseline (v1 pipeline)
- simdino efficembed (v1 pipeline)
- simdinov2 baseline (DINOv2 pipeline)
- simdinov2 efficembed (DINOv2 pipeline)

Fix any errors before submitting full jobs.

---

## Step 6 — Submit training jobs

```bash
cd ~/EfficDINO

# Exp 1 — SimDINOv2 baseline
sbatch scripts/slurm_simdino_baseline_v2.sh

# Exp 2 — EfficEmbedDINOv2
sbatch scripts/slurm_efficembed_v2.sh
```

Monitor:
```bash
squeue -u u27077
tail -f logs/simdinov2-baseline-<jobid>.out
tail -f logs/efficembed-v2-<jobid>.out
```

**Jobs auto-resume** if they hit the 48h wall time — just resubmit the same `sbatch` command.

---

## Reference

### Cluster
| Partition | Nodes | GPUs | Time limit |
|-----------|-------|------|------------|
| `kisski` | 24 (ggpu170–193) | 4× A100 80GB | 48h |
| `kisski-h100` | 8 (ggpu238–245) | H100 | 48h |

### Scripts
| Script | Pipeline | Purpose |
|--------|----------|---------|
| `slurm_simdino_baseline_v2.sh` | simdinov2 | Exp 1 full training |
| `slurm_efficembed_v2.sh` | simdinov2 | Exp 2 full training |
| `slurm_simdino_baseline.sh` | simdino v1 | Exp 1 (v1 reference) |
| `slurm_efficembed.sh` | simdino v1 | Exp 2 (v1 reference) |
| `smoke_test.sh` | both | 20-iteration sanity check |

### Checkpoints
- `checkpoints/simdinov2_vitb16_baseline/` — Exp 1 output
- `checkpoints/efficembed_vitb16/` — Exp 2 output

### WandB
- Project: `efficdino`
- Runs: `simdinov2_vitb16_baseline`, `efficembed_vitb16`
