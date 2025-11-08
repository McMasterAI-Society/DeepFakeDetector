# datautil/preproc_dataset.py
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from research.preprocess_pipeline import preprocess_all  # your module

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}

class PreprocDataset(Dataset):
    """
    Loads files from root/{real,fake}, calls preprocess_all(path),
    and attaches a numeric label: real=0, fake=1.
    """
    def __init__(self, root: str | Path):
        self.items: List[Tuple[str, int]] = []
        root = Path(root)
        for y, cls in enumerate(["real", "fake"]):
            d = root / cls
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in VALID_EXT:
                    self.items.append((str(f), y))
        if not self.items:
            raise RuntimeError(f"No images found under {root}/real or {root}/fake")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx) -> Dict:
        path, y = self.items[idx]
        out = preprocess_all(path)  # returns dict with vit/cnn_patches/fft_tensor/meta
        out["label"] = y            # attach label here
        return out


def collate(batch: List[Dict]) -> Dict:
    """
    Handles variable number of patches per image by keeping a list.
    Returns a dict ready for models:
      - vit: [B,3,S,S]
      - fft: [B,1,P,P]
      - patches: list of [N_i,3,P,P] tensors (per image)
      - labels: [B]
      - meta:   list of metadata dicts (per image)
    """
    labels = torch.tensor([b["label"] for b in batch]).long()
    vit = torch.stack([b["vit"] for b in batch])                 # [B,3,S,S]
    fft = torch.stack([b["fft_tensor"] for b in batch])          # [B,1,P,P]
    patches = [b["cnn_patches"] for b in batch]                  # list of [N_i,3,P,P]
    meta = [b["meta"] for b in batch]
    return {"vit": vit, "fft": fft, "patches": patches, "labels": labels, "meta": meta}
