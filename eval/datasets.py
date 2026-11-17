"""
Manifest-driven dataset loading for the evaluation harness.

A manifest is a CSV or JSONL file with one row per image:

    path,label,generator,source,format,width,height,prompt_id

* ``path`` is relative to the data root (``--data-root``; defaults to the
  manifest's own directory). Absolute paths are allowed but not portable.
* ``label`` is 0/1 or "real"/"fake".
* ``generator`` is the generator name for fakes and "real" for reals.
* ``source`` is the dataset/sub-folder the image came from.
* ``format`` is the decoded container format reported by PIL (jpeg, png, ...),
  not the file extension.
* ``prompt_id`` is optional and is used to group images generated from the
  same prompt so splits do not leak.

Any additional columns (``split``, ``variant``, ``orig_path``, ``sha256``...)
are preserved in ``Sample.extra``.

Only manifests are committed to git; images live in an HF dataset repo or on
Drive and are pointed to with ``--data-root``.

CLI::

    python -m eval.datasets scan-split  --split-dir <root>/benchmark_ten_percent/test --out eval/manifests/core.csv
    python -m eval.datasets scan-manual --manual-dir <root>/manual_gen_images --reals-dir <root>/OpenImagesV7/train --out m.csv
    python -m eval.datasets filter --manifest core.csv --exclude-generators sd15,sdxl --out heldout.csv
    python -m eval.datasets summary --manifest eval/manifests/core.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CORE_COLUMNS = ["path", "label", "generator", "source", "format", "width", "height", "prompt_id"]
REAL_GENERATOR = "real"

# Size buckets by the image's shorter side, in pixels.
SIZE_BUCKETS = [(0, 256, "<256"), (256, 512, "256-511"), (512, 1024, "512-1023"),
                (1024, 2048, "1024-2047"), (2048, 10 ** 9, ">=2048")]


@dataclass
class Sample:
    path: str
    label: int
    generator: str = ""
    source: str = ""
    format: str = ""
    width: int = 0
    height: int = 0
    prompt_id: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def size_bucket(self) -> str:
        return size_bucket(self.width, self.height)

    def to_row(self) -> Dict[str, Any]:
        row = {
            "path": self.path,
            "label": int(self.label),
            "generator": self.generator,
            "source": self.source,
            "format": self.format,
            "width": int(self.width),
            "height": int(self.height),
            "prompt_id": self.prompt_id,
        }
        row.update(self.extra)
        return row

    def get(self, key: str, default: str = "") -> Any:
        if key == "size_bucket":
            return self.size_bucket
        if hasattr(self, key) and key != "extra":
            return getattr(self, key)
        return self.extra.get(key, default)


def size_bucket(width: int, height: int) -> str:
    if not width or not height:
        return "unknown"
    side = min(int(width), int(height))
    for lo, hi, name in SIZE_BUCKETS:
        if lo <= side < hi:
            return name
    return "unknown"


def parse_label(value: Any) -> int:
    s = str(value).strip().lower()
    if s in {"1", "fake", "ai", "generated", "true"}:
        return 1
    if s in {"0", "real", "authentic", "false"}:
        return 0
    raise ValueError(f"Unrecognised label: {value!r}")


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def sample_from_row(row: Dict[str, Any]) -> Sample:
    row = {k: ("" if v is None else v) for k, v in row.items()}
    extra = {k: str(v) for k, v in row.items() if k not in CORE_COLUMNS}
    return Sample(
        path=str(row["path"]).replace("\\", "/"),
        label=parse_label(row["label"]),
        generator=str(row.get("generator", "")),
        source=str(row.get("source", "")),
        format=str(row.get("format", "")).lower(),
        width=_to_int(row.get("width", 0)),
        height=_to_int(row.get("height", 0)),
        prompt_id=str(row.get("prompt_id", "")),
        extra=extra,
    )


def file_sha256(path: os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class Manifest:
    """A list of samples plus the root their relative paths resolve against."""

    def __init__(self, samples: List[Sample], root: Optional[os.PathLike] = None,
                 source_path: Optional[os.PathLike] = None, sha256: Optional[str] = None):
        self.samples = samples
        self.root = Path(root) if root is not None else None
        self.source_path = Path(source_path) if source_path is not None else None
        self.sha256 = sha256

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def resolve(self, sample: Sample) -> Path:
        p = Path(sample.path)
        if p.is_absolute() or self.root is None:
            return p
        return self.root / p

    @property
    def labels(self) -> List[int]:
        return [s.label for s in self.samples]

    def info(self) -> Dict[str, Any]:
        return {
            "path": str(self.source_path) if self.source_path else None,
            "sha256": self.sha256,
            "n": len(self.samples),
            "n_real": sum(1 for s in self.samples if s.label == 0),
            "n_fake": sum(1 for s in self.samples if s.label == 1),
        }


def read_manifest(path: os.PathLike, root: Optional[os.PathLike] = None) -> Manifest:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    samples = [sample_from_row(r) for r in rows]
    return Manifest(samples, root=root if root is not None else path.parent,
                    source_path=path, sha256=file_sha256(path))


def write_manifest(samples: Sequence[Sample], path: os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [s.to_row() for s in samples]
    extra_cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in CORE_COLUMNS and k not in extra_cols:
                extra_cols.append(k)
    if path.suffix.lower() == ".jsonl":
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CORE_COLUMNS + extra_cols, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return path


def probe_image(path: os.PathLike):
    """Return (format, width, height) from the image header without decoding pixels."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            return (im.format or "").lower(), int(im.width), int(im.height)
    except Exception:
        return "", 0, 0


def _rel(path: Path, root: Optional[Path]) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def scan_split_layout(split_dir: os.PathLike, root: Optional[os.PathLike] = None,
                      split_name: Optional[str] = None, probe: bool = True,
                      hash_files: bool = False, max_per_source: Optional[int] = None,
                      seed: int = 0) -> List[Sample]:
    """Scan the AI-GenBench ``{split}/{real,fake}/{source}/`` layout.

    For fakes the source folder name is the generator. Images directly inside
    ``real/`` or ``fake/`` get source "unknown".
    """
    split_dir = Path(split_dir)
    root_p = Path(root) if root is not None else split_dir
    split_name = split_name if split_name is not None else split_dir.name
    rng = random.Random(seed)
    samples: List[Sample] = []
    for label, cls in enumerate(["real", "fake"]):
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        groups: Dict[str, List[Path]] = {}
        for entry in sorted(cls_dir.iterdir()):
            if entry.is_dir():
                groups[entry.name] = _list_images(entry)
            elif entry.is_file() and entry.suffix.lower() in IMG_EXTS:
                groups.setdefault("unknown", []).append(entry)
        for src, files in groups.items():
            if max_per_source is not None and len(files) > max_per_source:
                files = sorted(rng.sample(files, max_per_source))
            for f in files:
                fmt, w, h = probe_image(f) if probe else ("", 0, 0)
                extra = {"split": split_name}
                if hash_files:
                    extra["sha256"] = file_sha256(f)
                samples.append(Sample(
                    path=_rel(f, root_p), label=label,
                    generator=src if label == 1 else REAL_GENERATOR,
                    source=src, format=fmt, width=w, height=h, prompt_id="", extra=extra,
                ))
    return samples


def _load_manual_metadata(manual_dir: Path) -> Dict[str, Dict[str, str]]:
    """Optional metadata.{csv,jsonl} keyed by '<generator>/<filename>' or filename."""
    meta: Dict[str, Dict[str, str]] = {}
    for name in ("metadata.csv", "metadata.jsonl"):
        p = manual_dir / name
        if not p.exists():
            continue
        if p.suffix == ".jsonl":
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            with open(p, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        for r in rows:
            key = r.get("file") or r.get("path") or r.get("image_path") or r.get("filename")
            if not key:
                continue
            key = str(key).replace("\\", "/")
            meta[key] = {k: str(v) for k, v in r.items()}
            meta.setdefault(key.split("/")[-1], meta[key])
    return meta


def scan_manual_layout(manual_dir: os.PathLike, root: Optional[os.PathLike] = None,
                       reals_dir: Optional[os.PathLike] = None, n_reals: Optional[int] = None,
                       real_source: str = "openimages", probe: bool = True,
                       hash_files: bool = False, seed: int = 42) -> List[Sample]:
    """Scan the manual-gen layout ``images/<generator>/<prompt_id>.png``.

    ``prompt_id`` comes from ``metadata.{csv,jsonl}`` when present, else the
    filename stem (the convention used by ``DeepFakeDetector/manual-gen-images``).
    Reals can be added from ``reals_dir`` (searched recursively); by default as
    many reals as fakes are sampled. Reals get an empty prompt_id.
    """
    manual_dir = Path(manual_dir)
    images_dir = manual_dir / "images" if (manual_dir / "images").is_dir() else manual_dir
    root_p = Path(root) if root is not None else manual_dir
    meta = _load_manual_metadata(manual_dir)
    samples: List[Sample] = []
    for gen_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        for f in _list_images(gen_dir):
            m = meta.get(f"{gen_dir.name}/{f.name}") or meta.get(f.name) or {}
            fmt, w, h = probe_image(f) if probe else ("", 0, 0)
            extra = {"split": m.get("split", "")} if m.get("split") else {}
            if hash_files:
                extra["sha256"] = file_sha256(f)
            samples.append(Sample(
                path=_rel(f, root_p), label=1,
                generator=m.get("generator") or gen_dir.name, source="manual-gen",
                format=fmt, width=w, height=h,
                prompt_id=m.get("prompt_id") or f.stem, extra=extra,
            ))
    if reals_dir is not None:
        reals = sorted(p for p in Path(reals_dir).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
        k = len(samples) if n_reals is None else n_reals
        if k < len(reals):
            reals = sorted(random.Random(seed).sample(reals, k))
        for f in reals:
            fmt, w, h = probe_image(f) if probe else ("", 0, 0)
            extra = {"sha256": file_sha256(f)} if hash_files else {}
            samples.append(Sample(
                path=_rel(f, root_p), label=0, generator=REAL_GENERATOR, source=real_source,
                format=fmt, width=w, height=h, prompt_id="", extra=extra,
            ))
    return samples


def filter_samples(samples: Iterable[Sample], include_generators: Optional[Sequence[str]] = None,
                   exclude_generators: Optional[Sequence[str]] = None,
                   split: Optional[str] = None) -> List[Sample]:
    """Filter fakes by generator (reals are always kept) and optionally by split."""
    inc = {g.lower() for g in include_generators} if include_generators else None
    exc = {g.lower() for g in exclude_generators} if exclude_generators else set()
    out = []
    for s in samples:
        if split is not None and s.extra.get("split", "") != split:
            continue
        if s.label == 1:
            g = s.generator.lower()
            if inc is not None and g not in inc:
                continue
            if g in exc:
                continue
        out.append(s)
    return out


def summarize(samples: Sequence[Sample]) -> Dict[str, Any]:
    by = lambda key: dict(Counter(f"{'fake' if s.label else 'real'}:{s.get(key)}" for s in samples))
    return {
        "n": len(samples),
        "n_real": sum(1 for s in samples if s.label == 0),
        "n_fake": sum(1 for s in samples if s.label == 1),
        "by_generator": by("generator"),
        "by_format": by("format"),
        "by_size_bucket": by("size_bucket"),
    }


def _csv_list(s: Optional[str]) -> Optional[List[str]]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.datasets", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("scan-split", help="scan {split}/{real,fake}/{source}/")
    a.add_argument("--split-dir", required=True)
    a.add_argument("--root", help="paths are written relative to this (default: --split-dir)")
    a.add_argument("--split-name")
    a.add_argument("--max-per-source", type=int)
    a.add_argument("--hash", action="store_true", help="add a sha256 column per image")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", required=True)

    b = sub.add_parser("scan-manual", help="scan images/<generator>/<prompt_id>.png")
    b.add_argument("--manual-dir", required=True)
    b.add_argument("--reals-dir")
    b.add_argument("--n-reals", type=int)
    b.add_argument("--real-source", default="openimages")
    b.add_argument("--root", help="paths are written relative to this (default: --manual-dir)")
    b.add_argument("--hash", action="store_true")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--out", required=True)

    c = sub.add_parser("filter", help="filter fakes by generator / split")
    c.add_argument("--manifest", required=True)
    c.add_argument("--include-generators")
    c.add_argument("--exclude-generators")
    c.add_argument("--split")
    c.add_argument("--out", required=True)

    d = sub.add_parser("summary", help="print counts per generator/format/size")
    d.add_argument("--manifest", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "scan-split":
        samples = scan_split_layout(args.split_dir, root=args.root, split_name=args.split_name,
                                    hash_files=args.hash, max_per_source=args.max_per_source,
                                    seed=args.seed)
    elif args.cmd == "scan-manual":
        samples = scan_manual_layout(args.manual_dir, root=args.root, reals_dir=args.reals_dir,
                                     n_reals=args.n_reals, real_source=args.real_source,
                                     hash_files=args.hash, seed=args.seed)
    elif args.cmd == "filter":
        m = read_manifest(args.manifest)
        samples = filter_samples(m.samples, _csv_list(args.include_generators),
                                 _csv_list(args.exclude_generators), args.split)
    else:
        m = read_manifest(args.manifest)
        print(json.dumps(summarize(m.samples), indent=2))
        return 0
    out = write_manifest(samples, args.out)
    print(json.dumps({"out": str(out), **summarize(samples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
