"""
Bias-probe suite: the same images under matched formats and sizes.

If a detector separates real from fake because reals are JPEG and fakes are
PNG (or because of resolution), its AUROC drops sharply once those cues are
equalised. Variants built from a source manifest:

    original              unmodified
    real_png              reals decoded and re-saved as PNG; fakes unchanged
    fake_jpeg_q{95,85,75} fakes JPEG-compressed at quality Q; reals unchanged
    resized_{S}           both classes: shorter side -> S, center crop SxS, saved as PNG
    matched_{S}_jpeg_q85  both classes: resized as above and saved as JPEG Q85
                          (format, quality and size fully matched)

Generated images go under ``<data-root>/<cache-subdir>/`` (gitignored, never
committed) and the manifest stores paths relative to ``--data-root``, so the
committed manifest stays valid wherever the data is mounted. Generation is
deterministic given the source manifest and options.

    python -m eval.bias_probe build --manifest eval/manifests/core.csv --data-root /content/data \\
        --out eval/manifests/bias-probe.csv --max-per-class 1000

``python -m eval.run_eval --suite bias-probe`` then reports AUROC per variant
and the drop vs ``original`` (see ``report``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from eval.datasets import Manifest, Sample, read_manifest, write_manifest
from eval import metrics as M

DEFAULT_QUALITIES = (95, 85, 75)
DEFAULT_SIZE = 512
DEFAULT_SUBDIR = "_bias_probe"


def _resize_center_crop(img, size: int):
    from PIL import Image

    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def _save(img, path: Path, fmt: str, quality: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jpeg":
        # 4:2:0 subsampling is what most cameras and social platforms use.
        img.save(path, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    else:
        img.save(path, format="PNG", optimize=False)


def variant_names(qualities: Sequence[int] = DEFAULT_QUALITIES, size: int = DEFAULT_SIZE,
                  matched_quality: int = 85) -> List[str]:
    return (["original", "real_png"] + [f"fake_jpeg_q{q}" for q in qualities]
            + [f"resized_{size}", f"matched_{size}_jpeg_q{matched_quality}"])


def _stable_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def build(manifest: Manifest, data_root: Path, out_manifest: Path,
          qualities: Sequence[int] = DEFAULT_QUALITIES, size: int = DEFAULT_SIZE,
          matched_quality: int = 85, max_per_class: Optional[int] = None, seed: int = 0,
          subdir: str = DEFAULT_SUBDIR, overwrite: bool = False) -> List[Sample]:
    """Generate variant images and write the bias-probe manifest."""
    from PIL import Image

    data_root = Path(data_root)
    cache = data_root / subdir
    samples = list(manifest.samples)
    if max_per_class is not None:
        rng = random.Random(seed)
        picked = []
        for lbl in (0, 1):
            cls = [s for s in samples if s.label == lbl]
            if len(cls) > max_per_class:
                cls = rng.sample(cls, max_per_class)
            picked.extend(sorted(cls, key=lambda s: s.path))
        samples = picked

    rows: List[Sample] = []

    def add(src: Sample, variant: str, rel: str, fmt: str, w: int, h: int):
        extra = dict(src.extra)
        extra.update(variant=variant, orig_path=src.path)
        rows.append(Sample(path=rel, label=src.label, generator=src.generator, source=src.source,
                           format=fmt, width=w, height=h, prompt_id=src.prompt_id, extra=extra))

    for s in samples:
        src_path = manifest.resolve(s)
        uid = _stable_id(s.path)
        cls = "fake" if s.label else "real"
        add(s, "original", s.path, s.format, s.width, s.height)
        img = None

        def get_img():
            nonlocal img
            if img is None:
                with Image.open(src_path) as im:
                    img = im.convert("RGB")
            return img

        def emit(variant: str, transform, fmt: str, quality: Optional[int] = None):
            ext = "jpg" if fmt == "jpeg" else "png"
            rel = f"{subdir}/{variant}/{cls}/{uid}.{ext}"
            out = cache / variant / cls / f"{uid}.{ext}"
            if overwrite or not out.exists():
                _save(transform(get_img()), out, fmt, quality)
            if transform is _identity:
                w, h = (s.width, s.height) if s.width else get_img().size
            else:
                w = h = size
            add(s, variant, rel, fmt, w, h)

        # Format equalisation on one class only; the other class keeps its original file.
        if s.label == 0:
            emit("real_png", _identity, "png")
        else:
            add(s, "real_png", s.path, s.format, s.width, s.height)
        for q in qualities:
            if s.label == 1:
                emit(f"fake_jpeg_q{q}", _identity, "jpeg", q)
            else:
                add(s, f"fake_jpeg_q{q}", s.path, s.format, s.width, s.height)
        emit(f"resized_{size}", lambda im: _resize_center_crop(im, size), "png")
        emit(f"matched_{size}_jpeg_q{matched_quality}", lambda im: _resize_center_crop(im, size),
             "jpeg", matched_quality)

    write_manifest(rows, out_manifest)
    return rows


def _identity(im):
    return im


def report(samples: Sequence[Sample], prob_fake: np.ndarray, n_boot: int = 1000,
           seed: int = 0) -> Dict[str, Any]:
    """AUROC per variant and paired drop vs ``original`` (DeLong on matched images)."""
    y_all = np.array([s.label for s in samples], dtype=int)
    p_all = np.asarray(prob_fake, dtype=float)
    variants: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        variants.setdefault(s.extra.get("variant", "original"), []).append(i)
    if "original" not in variants:
        raise ValueError("bias-probe manifest has no 'original' variant rows")

    def keyed(idx):
        return {samples[i].extra.get("orig_path", samples[i].path): i for i in idx
                if np.isfinite(p_all[i])}

    base = keyed(variants["original"])
    out: Dict[str, Any] = {"variants": {}}
    base_auc = None
    for v, idx in variants.items():
        k = keyed(idx)
        common = sorted(set(k) & set(base))
        iv = np.array([k[c] for c in common], dtype=int)
        ib = np.array([base[c] for c in common], dtype=int)
        y = y_all[iv]
        entry: Dict[str, Any] = {"n": int(len(iv)), "auroc": M.auroc(y, p_all[iv]),
                                 "ci95": M.bootstrap_ci(y, p_all[iv], n_boot=n_boot, seed=seed,
                                                        metrics=["auroc"])["auroc"],
                                 "mean_prob_fake_real": _mean(p_all[iv][y == 0]),
                                 "mean_prob_fake_fake": _mean(p_all[iv][y == 1])}
        if v == "original":
            base_auc = entry["auroc"]
        elif len(iv):
            t = M.delong_test(y, p_all[ib], p_all[iv])
            if t:
                entry["auroc_original_paired"] = t["auroc_1"]
                entry["auroc_drop"] = t["diff"]
                entry["delong_p"] = t["p_value"]
        out["variants"][v] = entry
    drops = [(e["auroc_drop"], v) for v, e in out["variants"].items() if e.get("auroc_drop") is not None]
    worst = max(drops) if drops else (None, None)
    out["auroc_original"] = base_auc
    out["max_auroc_drop"], out["worst_variant"] = worst
    return out


def _mean(x) -> Optional[float]:
    return float(np.mean(x)) if len(x) else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.bias_probe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--manifest", required=True, help="source manifest, e.g. eval/manifests/core.csv")
    b.add_argument("--data-root", required=True)
    b.add_argument("--out", default="eval/manifests/bias-probe.csv")
    b.add_argument("--size", type=int, default=DEFAULT_SIZE)
    b.add_argument("--qualities", default=",".join(map(str, DEFAULT_QUALITIES)))
    b.add_argument("--matched-quality", type=int, default=85)
    b.add_argument("--max-per-class", type=int)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--subdir", default=DEFAULT_SUBDIR)
    b.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    m = read_manifest(args.manifest, root=args.data_root)
    rows = build(m, Path(args.data_root), Path(args.out),
                 qualities=[int(q) for q in args.qualities.split(",") if q],
                 size=args.size, matched_quality=args.matched_quality,
                 max_per_class=args.max_per_class, seed=args.seed, subdir=args.subdir,
                 overwrite=args.overwrite)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.extra["variant"]] = counts.get(r.extra["variant"], 0) + 1
    print(json.dumps({"out": args.out, "rows": len(rows), "per_variant": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
