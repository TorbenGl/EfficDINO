"""
Download ILSVRC/imagenet-1k from HuggingFace Hub.

Resumable — re-running after interruption continues where it left off.

Usage:
    python scripts/download_imagenet.py --save_path /path/to/imagenet-1k
    python scripts/download_imagenet.py --save_path /path/to/imagenet-1k --max_workers 2
    python scripts/download_imagenet.py  # uses IMAGENET_SAVE_PATH env var or ./imagenet-1k

Requires:
    HF_TOKEN env var set to a HuggingFace token with access to ILSVRC/imagenet-1k
    pip install huggingface_hub
"""

import argparse
import logging
import os
import sys

# Disable hf_transfer (fast-download lib that benchmarks on startup) and
# any disk-speed benchmarking done by the datasets library.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_DATASETS_DISABLE_DISK_SPACE_USAGE_STATS", "1")


# -------- LOGGING --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger()


DATASET_NAME = "ILSVRC/imagenet-1k"


def parse_args():
    parser = argparse.ArgumentParser(description="Download ILSVRC/imagenet-1k from HuggingFace Hub")
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.getenv("IMAGENET_SAVE_PATH", "./imagenet-1k"),
        help="Directory to save the dataset (default: $IMAGENET_SAVE_PATH or ./imagenet-1k)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace token (default: $HF_TOKEN env var)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Max concurrent download threads (default: 2, lower = less CPU/RAM)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Add file handler now that we know save_path
    os.makedirs(args.save_path, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.save_path, "download.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info(f"Save path: {args.save_path}")
    logger.info(f"Max workers: {args.max_workers}")

    if args.token is None:
        logger.error("HF_TOKEN not set — pass --token or set HF_TOKEN env var")
        sys.exit(1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed — run: pip install huggingface_hub")
        sys.exit(1)

    logger.info("Starting resumable download (re-run to continue if interrupted)...")

    snapshot_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        local_dir=args.save_path,
        token=args.token,
        resume_download=True,
        max_workers=args.max_workers,
    )

    logger.info(f"Download complete: {args.save_path}")
    logger.info("To use in training: set HF_DATA_PATH=<save_path> and use --dataset_type hf_imagenet")


if __name__ == "__main__":
    main()
