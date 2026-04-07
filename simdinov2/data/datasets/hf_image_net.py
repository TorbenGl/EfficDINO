import torch.utils.data


class HFImageNetDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace imagenet-1k snapshot_download dataset.

    Dataset string format (for simdinov2 loaders.py):
        HFImageNet:split=TRAIN:root=/path/to/hf-snapshot
        HFImageNet:split=VAL:root=/path/to/hf-snapshot

    The snapshot must have been downloaded with:
        python scripts/download_imagenet.py --save_path /path/to/hf-snapshot
    """

    _SPLIT_MAP = {
        "TRAIN": "train",
        "VAL": "validation",
        "TEST": "test",
    }

    def __init__(self, root, split, transform=None, target_transform=None):
        from datasets import load_dataset

        hf_split = self._SPLIT_MAP.get(split, split)
        self.dataset = load_dataset(root, split=hf_split, trust_remote_code=False)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"].convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = sample["label"]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label
