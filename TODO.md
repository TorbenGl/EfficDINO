# EfficDINO — Known Issues / TODO

Minor issues to fix before full training run.

---

- [ ] `HF_DATA_PATH` must be a full absolute path (no `~`) — `datasets` library does not expand tildes; throws `HFValidationError`. Add validation or note in scripts.
- [ ] HF Arrow cache was writing to `~/.cache/huggingface/datasets/` hitting home quota — fixed to write to `${HF_DATA_PATH}/.hf_cache/`, but cache dir should be confirmed writable before training starts.
- [ ] `simdino/main_dino.py:330` — `shutil.copyfile("main_dino.py", ...)` used hardcoded relative path; fixed to use `Path(__file__)` but needs smoke test confirmation.
- [ ] Smoke test runs 4 GPUs by default (`NUM_GPUS=4`) — single-GPU nodes need `NUM_GPUS=1` override; should be documented more prominently.
- [ ] RUNBOOK.md Step 4: replace `/path/to/data/imagenet-1k` placeholder with a note to use the actual absolute path.
- [ ] `--compile True` (default) crashes with `InductorError` on some GPU architectures (seen on H200) — smoke test disabled it with `--compile False`; decide whether to keep compile on or off for full training runs and update `run_or_resume.sh` accordingly.
