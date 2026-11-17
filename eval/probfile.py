"""
Probability files: per-image predictions that can be produced on Colab and
scored or re-fused offline.

Format (CSV, or NPZ with one array per column):

    path,label,generator,source,format,width,height,prompt_id,[extras...],prob_fake,prob.<submodel>...,error

* ``prob_fake`` is the candidate's final P(fake).
* ``prob.<submodel>`` columns (e.g. ``prob.cnn-transfer``) hold per-submodel
  P(fake) when available; ``eval/fusion_refit.py`` fits on these.
* ``error`` is non-empty for images that failed; their probabilities are NaN.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from eval.datasets import CORE_COLUMNS, Sample, sample_from_row

SUB_PREFIX = "prob."
PROB_COL = "prob_fake"


class ProbTable:
    """Column-oriented view of a probability file."""

    def __init__(self, columns: Dict[str, np.ndarray], source_path: Optional[Path] = None):
        self.columns = columns
        self.source_path = source_path
        n = {len(v) for v in columns.values()}
        if len(n) > 1:
            raise ValueError(f"ragged probability file columns: {n}")
        self.n = n.pop() if n else 0

    def __len__(self) -> int:
        return self.n

    def has(self, name: str) -> bool:
        return name in self.columns

    def floats(self, name: str) -> np.ndarray:
        return np.array([_to_float(v) for v in self.columns[name]], dtype=float)

    def strings(self, name: str) -> List[str]:
        if name not in self.columns:
            return [""] * self.n
        return ["" if v is None else str(v) for v in self.columns[name]]

    @property
    def submodels(self) -> List[str]:
        return [c[len(SUB_PREFIX):] for c in self.columns if c.startswith(SUB_PREFIX)]

    def samples(self) -> List[Sample]:
        meta_cols = [c for c in self.columns
                     if c != PROB_COL and c != "error" and not c.startswith(SUB_PREFIX)]
        out = []
        for i in range(self.n):
            row = {c: self.columns[c][i] for c in meta_cols}
            out.append(sample_from_row(row))
        return out

    @property
    def labels(self) -> np.ndarray:
        return np.array([s.label for s in self.samples()], dtype=int)


def _to_float(v) -> float:
    try:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return math.nan
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def read_probs(path: os.PathLike) -> ProbTable:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            cols = {k: z[k] for k in z.files}
        cols = {k: (v.astype(str) if v.dtype.kind in "US" else v) for k, v in cols.items()}
        return ProbTable(cols, path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return ProbTable({}, path)
    names = list(rows[0].keys())
    cols = {k: np.array([r.get(k, "") for r in rows], dtype=object) for k in names}
    return ProbTable(cols, path)


def write_probs(path: os.PathLike, samples: Sequence[Sample], prob_fake: np.ndarray,
                sub_probs: Optional[Dict[str, np.ndarray]] = None,
                errors: Optional[Dict[int, str]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub_probs = sub_probs or {}
    errors = errors or {}
    rows = [s.to_row() for s in samples]
    extra_cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in CORE_COLUMNS and k not in extra_cols:
                extra_cols.append(k)
    sub_cols = [SUB_PREFIX + k for k in sub_probs]
    header = CORE_COLUMNS + extra_cols + [PROB_COL] + sub_cols + ["error"]
    for i, r in enumerate(rows):
        r[PROB_COL] = _fmt(prob_fake[i])
        for k, arr in sub_probs.items():
            r[SUB_PREFIX + k] = _fmt(arr[i])
        r["error"] = errors.get(i, "")
    if path.suffix.lower() == ".npz":
        arrays = {}
        for c in header:
            vals = [r.get(c, "") for r in rows]
            if c == PROB_COL or c in sub_cols:
                arrays[c] = np.array([_to_float(v) for v in vals], dtype=float)
            else:
                arrays[c] = np.array([str(v) for v in vals], dtype=str)
        np.savez_compressed(path, **arrays)
    else:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in header})
    return path


def _fmt(v) -> str:
    v = float(v)
    return "" if math.isnan(v) else repr(v)


def align_to_samples(table: ProbTable, samples: Sequence[Sample]) -> np.ndarray:
    """Row index into ``table`` for each sample (-1 if missing).

    Matches on the manifest-relative ``path`` first, then on a unique basename.
    """
    paths = table.strings("path")
    norm = [p.replace("\\", "/") for p in paths]
    by_path = {p: i for i, p in enumerate(norm)}
    by_base: Dict[str, int] = {}
    dup = set()
    for i, p in enumerate(norm):
        b = p.rsplit("/", 1)[-1]
        if b in by_base:
            dup.add(b)
        by_base[b] = i
    out = np.full(len(samples), -1, dtype=int)
    for j, s in enumerate(samples):
        p = s.path.replace("\\", "/")
        if p in by_path:
            out[j] = by_path[p]
            continue
        b = p.rsplit("/", 1)[-1]
        if b in by_base and b not in dup:
            out[j] = by_base[b]
    return out
