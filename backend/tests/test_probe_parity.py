"""
Parity of the served foundation probe against the experiment's predictions.

Runs only when the real backbones and experiment artifacts are available:

    PROBE_PARITY_ROOT       experiment root containing
                              experiments/foundation/weights/<key>.fp16.pt
                              experiments/foundation/best/{config.json, head.npz}
                              experiments/foundation/results/clipdino_both_coco/*.preds.csv
                              heldout/{core-test-sq, core-test}/...
    PROBE_PARITY_HEAD_DIR   optional: head dir to test instead of best/ (e.g. an export)
    PROBE_PARITY_RESULTS    optional: results dir whose preds belong to that head
    PROBE_PARITY_N          images per check (default 50)

Two checks, tolerance 0.005 on prob_fake:
  1. core-test-sq images (already canonicalised offline; predicted with
     canonicalize=False, as in the experiment) vs core-test-sq__none preds.
  2. raw core-test images through the full predict() path (EXIF, 512/JPEG-q90
     canonicalisation, crop) vs core-test-norm__none preds, whose images were
     canonicalised offline with the same steps.
"""

import csv
import json
import os
from pathlib import Path

import pytest

ROOT = os.environ.get("PROBE_PARITY_ROOT")
N = int(os.environ.get("PROBE_PARITY_N", "50"))
TOL = 0.005

pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "experiments" / "foundation" / "weights").is_dir(),
    reason="PROBE_PARITY_ROOT with real backbones/artifacts not available",
)


def _paths():
    root = Path(ROOT)
    fdir = root / "experiments" / "foundation"
    head_dir = Path(os.environ.get("PROBE_PARITY_HEAD_DIR", fdir / "best"))
    results = Path(os.environ.get("PROBE_PARITY_RESULTS", fdir / "results" / "clipdino_both_coco"))
    return root, fdir, head_dir, results


@pytest.fixture(scope="module")
def probe():
    from app.core.config import settings
    from app.models.wrappers.foundation_probe_wrapper import FoundationProbeWrapper

    root, fdir, head_dir, _ = _paths()
    old = settings.PROBE_BACKBONE_DIR
    settings.PROBE_BACKBONE_DIR = str(fdir / "weights")
    try:
        w = FoundationProbeWrapper(
            "parity/probe", json.loads((head_dir / "config.json").read_text()), str(head_dir)
        )
        w.load()
    finally:
        settings.PROBE_BACKBONE_DIR = old
    return w


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _evenly(rows, n):
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def test_parity_core_test_sq(probe):
    root, _, _, results = _paths()
    rows = _evenly(_rows(results / "core-test-sq__none.preds.csv"), N)
    diffs = []
    for r in rows:
        data = (root / "heldout" / "core-test-sq" / r["path"]).read_bytes()
        p = probe.predict(image_bytes=data, canonicalize=False)["prob_fake"]
        diffs.append((abs(p - float(r["prob_fake"])), r["path"]))
    worst = max(diffs)
    assert worst[0] <= TOL, f"max |dp| {worst[0]:.4f} at {worst[1]}"


def test_parity_full_path_raw_core_test(probe):
    root, _, _, results = _paths()
    norm = {r["path"]: float(r["prob_fake"]) for r in _rows(results / "core-test-norm__none.preds.csv")}
    raw = _evenly(_rows(root / "heldout" / "core-test" / "manifest.csv"), N)
    diffs = []
    for r in raw:
        key = r["path"].rsplit(".", 1)[0] + ".jpg"
        if key not in norm:
            continue
        data = (root / "heldout" / "core-test" / r["path"]).read_bytes()
        p = probe.predict(image_bytes=data)["prob_fake"]
        diffs.append((abs(p - norm[key]), r["path"]))
    assert len(diffs) >= N // 2
    worst = max(diffs)
    assert worst[0] <= TOL, f"max |dp| {worst[0]:.4f} at {worst[1]}"
