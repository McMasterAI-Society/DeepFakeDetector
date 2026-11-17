import json

import numpy as np
from PIL import Image

from eval import bias_probe
from eval.datasets import (filter_samples, read_manifest, scan_manual_layout, scan_split_layout,
                           size_bucket, write_manifest)
from eval.tests.conftest import make_image


def test_scan_split_and_roundtrip(split_tree, tmp_path):
    root, split = split_tree
    samples = scan_split_layout(split, root=root)
    assert len(samples) == 20
    reals = [s for s in samples if s.label == 0]
    fakes = [s for s in samples if s.label == 1]
    assert {s.source for s in reals} == {"coco", "laion"} and {s.generator for s in reals} == {"real"}
    assert {s.generator for s in fakes} == {"sd15", "midjourney", "flux"}
    assert all(s.format == "jpeg" and (s.width, s.height) == (80, 60) for s in reals)
    assert all(s.format == "png" for s in fakes)
    assert all(not s.path.startswith("/") and "\\" not in s.path for s in samples)
    assert samples[0].extra["split"] == "test"

    for ext in ("csv", "jsonl"):
        out = write_manifest(samples, tmp_path / f"m.{ext}")
        m = read_manifest(out, root=root)
        assert [s.to_row() for s in m.samples] == [s.to_row() for s in samples]
        assert m.resolve(m.samples[0]).exists()
        assert len(m.sha256) == 64

    held = filter_samples(samples, exclude_generators=["sd15"])
    assert len(held) == 16 and all(s.generator != "sd15" for s in held)


def test_scan_manual_layout(tmp_path):
    md = tmp_path / "manual_gen_images"
    for gen in ("dalle", "nano-banana"):
        for pid in ("p001", "p002"):
            make_image(md / "images" / gen / f"{pid}.png", (32, 32), "PNG")
    (md / "prompts").mkdir()
    reals = tmp_path / "openimages"
    for i in range(10):
        make_image(reals / "sub" / f"{i}.jpg", (40, 30), "JPEG", seed=i)
    samples = scan_manual_layout(md, root=tmp_path, reals_dir=reals, seed=0)
    fakes = [s for s in samples if s.label == 1]
    assert len(fakes) == 4 and {s.prompt_id for s in fakes} == {"p001", "p002"}
    assert {s.generator for s in fakes} == {"dalle", "nano-banana"}
    assert sum(1 for s in samples if s.label == 0) == 4  # matched to #fakes
    # metadata overrides the filename-stem prompt id
    (md / "metadata.csv").write_text("file,prompt_id\ndalle/p001.png,X9\n", encoding="utf-8")
    s2 = scan_manual_layout(md, root=tmp_path)
    assert {s.prompt_id for s in s2 if s.generator == "dalle"} == {"X9", "p002"}


def test_size_bucket():
    assert size_bucket(100, 2000) == "<256"
    assert size_bucket(512, 512) == "512-1023"
    assert size_bucket(0, 10) == "unknown"


def test_bias_probe_build_and_report(split_tree, tmp_path):
    root, split = split_tree
    src = write_manifest(scan_split_layout(split, root=root), tmp_path / "core.csv")
    m = read_manifest(src, root=root)
    out = tmp_path / "bias-probe.csv"
    rows = bias_probe.build(m, root, out, qualities=(95, 75), size=48, matched_quality=85)
    names = bias_probe.variant_names((95, 75), 48, 85)
    probe = read_manifest(out, root=root)
    assert len(probe) == len(m) * len(names) == len(rows)
    by_v = {}
    for s in probe:
        by_v.setdefault(s.extra["variant"], []).append(s)
    assert set(by_v) == set(names)

    def fmt_size(s):
        with Image.open(probe.resolve(s)) as im:
            return im.format, im.size

    for s in by_v["real_png"]:
        f, _ = fmt_size(s)
        assert f == "PNG"  # fixture fakes are PNG already
        if s.label == 0:
            assert s.path.startswith("_bias_probe/") and s.format == "png"
    for s in by_v["fake_jpeg_q75"]:
        f, _ = fmt_size(s)
        assert f == "JPEG"  # fixture reals are JPEG already
        assert (s.path == s.extra["orig_path"]) == (s.label == 0)
    for s in by_v["matched_48_jpeg_q85"]:
        assert fmt_size(s) == ("JPEG", (48, 48)) and (s.width, s.height) == (48, 48)
    for s in by_v["resized_48"]:
        assert fmt_size(s) == ("PNG", (48, 48))
    # originals untouched, generated files only under the cache subdir
    assert all(s.path == s.extra["orig_path"] for s in by_v["original"])
    assert all(p.suffix in {".png", ".jpg"} for p in (root / "_bias_probe").rglob("*.*"))

    # A pure format detector: perfect on original, collapses once formats match.
    p = np.array([1.0 if s.format == "png" else 0.0 for s in probe], dtype=float)
    rep = bias_probe.report(probe.samples, p, n_boot=50)
    assert rep["auroc_original"] == 1.0
    assert rep["variants"]["matched_48_jpeg_q85"]["auroc"] == 0.5
    assert rep["variants"]["fake_jpeg_q75"]["auroc_drop"] == 0.5
    assert rep["max_auroc_drop"] == 0.5
    json.dumps(rep)

    # deterministic + idempotent
    rows2 = bias_probe.build(m, root, tmp_path / "again.csv", qualities=(95, 75), size=48)
    assert [r.path for r in rows2] == [r.path for r in rows]
