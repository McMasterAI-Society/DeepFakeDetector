"""Shared synthetic fixtures. No network, no model weights."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_image(path: Path, size=(64, 48), fmt="PNG", seed=0, quality=95):
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"quality": quality} if fmt == "JPEG" else {}
    Image.fromarray(arr).save(path, format=fmt, **kw)
    return path


@pytest.fixture
def split_tree(tmp_path):
    """benchmark-style {split}/{real,fake}/{source}/ with JPEG reals and PNG fakes."""
    root = tmp_path / "data"
    split = root / "benchmark_ten_percent" / "test"
    k = 0
    for src in ("coco", "laion"):
        for i in range(4):
            make_image(split / "real" / src / f"r{i}.jpg", (80, 60), "JPEG", seed=k); k += 1
    for gen in ("sd15", "midjourney", "flux"):
        for i in range(4):
            make_image(split / "fake" / gen / f"f{i}.png", (64, 64), "PNG", seed=k); k += 1
    return root, split


@pytest.fixture
def synthetic_scores():
    rng = np.random.default_rng(123)
    n_pos, n_neg = 300, 400
    y = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
    s = np.r_[rng.normal(1.0, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)]
    p = 1 / (1 + np.exp(-s))
    # introduce ties
    p = np.round(p, 2)
    return y, p
