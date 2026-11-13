"""
Candidate adapters: anything that turns a list of samples into P(fake).

Candidate specs accepted by ``load_candidate`` / ``--candidate``:

``deployed`` or ``deployed@<fusion_repo_id>``
    The production stack in-process: ``backend/app`` ModelRegistry downloads
    the fusion repo and its submodels from HF, then every image goes through
    ``wrapper.predict(image_bytes=...)`` and the fusion wrapper exactly as in
    ``/predict``. Per-submodel probabilities are recorded as well.

``submodel:<name>`` or ``submodel:<name>@<fusion_repo_id>``
    A single backend submodel (e.g. ``submodel:cnn-transfer``), resolved
    through the fusion config's ``submodels`` list. ``submodel:<org>/<repo>``
    loads that repo directly.

``<dir containing fusion_logreg.pkl>``
    A local fusion export (e.g. from ``eval/fusion_refit.py``). With
    ``--probs <file>`` it is applied offline to that file's ``prob.<name>``
    columns; without, the deployed submodels run in-process and this fusion
    replaces the deployed one.

``<file>.csv`` / ``<file>.npz`` [``#<column>``]
    Precomputed probabilities (e.g. from Colab). Defaults to the ``prob_fake``
    column; ``file.csv#cnn-transfer`` scores the ``prob.cnn-transfer`` column.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from eval.datasets import Sample
from eval.probfile import PROB_COL, SUB_PREFIX, ProbTable, align_to_samples, read_probs

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
DEFAULT_CACHE = REPO_ROOT / "eval" / ".cache"
DEFAULT_FUSION_REPO = "DeepFakeDetector/fusion-logreg-final"

Resolver = Callable[[Sample], Path]


@dataclass
class Predictions:
    prob_fake: np.ndarray
    sub_probs: Dict[str, np.ndarray] = field(default_factory=dict)
    errors: Dict[int, str] = field(default_factory=dict)
    seconds: float = 0.0


class Candidate(ABC):
    name: str = "candidate"
    threshold: float = 0.5
    threshold_override: Optional[float] = None

    @property
    def effective_threshold(self) -> float:
        return self.threshold if self.threshold_override is None else float(self.threshold_override)

    @abstractmethod
    def predict(self, samples: Sequence[Sample], resolve: Resolver) -> Predictions:
        ...

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "type": type(self).__name__, "threshold": self.effective_threshold}


def safe_name(spec: str) -> str:
    s = spec.rstrip("/\\")
    p = Path(s.split("#")[0])
    if p.suffix.lower() in {".csv", ".npz"}:
        base = p.stem + ("-" + s.split("#", 1)[1] if "#" in s else "")
    elif p.exists() and p.is_dir():
        base = p.name
    else:
        base = s
    return re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-") or "candidate"


# ---------------------------------------------------------------------------
# Backend (in-process)
# ---------------------------------------------------------------------------

def import_backend():
    """Make ``backend/app`` importable and point its HF cache somewhere writable."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    if not os.path.isdir("/data"):
        os.environ.setdefault("HF_CACHE_DIR", str(DEFAULT_CACHE / "hf"))
        os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE / "hf_home"))
    import app  # noqa: F401
    return app


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: Dict[str, Any] = {}

    def runner():
        try:
            box["v"] = asyncio.run(coro)
        except BaseException as e:  # pragma: no cover - propagated below
            box["e"] = e

    t = threading.Thread(target=runner)
    t.start()
    t.join()
    if "e" in box:
        raise box["e"]
    return box.get("v")


def _submodel_repo_ids(fusion_config: Dict[str, Any]) -> List[str]:
    out = []
    for entry in fusion_config.get("submodels", []):
        out.append(entry["repo_id"] if isinstance(entry, dict) else str(entry))
    return out


class _ImageLoopMixin:
    """Run a per-image function with de-duplication by resolved path."""

    def _loop(self, samples, resolve, fn, sub_names: Sequence[str]) -> Predictions:
        t0 = time.perf_counter()
        n = len(samples)
        prob = np.full(n, np.nan)
        subs = {k: np.full(n, np.nan) for k in sub_names}
        errors: Dict[int, str] = {}
        cache: Dict[str, Any] = {}
        progress = _progress(n, getattr(self, "name", "candidate"))
        for i, s in enumerate(samples):
            key = str(resolve(s))
            if key not in cache:
                try:
                    cache[key] = fn(Path(key).read_bytes())
                except Exception as e:  # keep going; failures are reported
                    cache[key] = e
            res = cache[key]
            if isinstance(res, Exception):
                errors[i] = f"{type(res).__name__}: {res}"
            else:
                prob[i] = res[0]
                for k, v in res[1].items():
                    if k in subs:
                        subs[k][i] = v
            progress(i + 1)
        return Predictions(prob, subs, errors, time.perf_counter() - t0)


def _progress(total: int, label: str):
    try:
        from tqdm.auto import tqdm  # type: ignore

        bar = tqdm(total=total, desc=label, mininterval=2.0)

        def step(i):
            bar.update(i - bar.n)
            if i == total:
                bar.close()
        return step
    except Exception:
        def step(i):
            if i == total or i % 500 == 0:
                print(f"[{label}] {i}/{total}", file=sys.stderr)
        return step


class BackendStackCandidate(_ImageLoopMixin, Candidate):
    """Deployed backend: all submodels + fusion, loaded via ModelRegistry."""

    def __init__(self, fusion_repo: Optional[str] = None, fusion_dir: Optional[Path] = None,
                 name: str = "deployed"):
        self.fusion_repo = fusion_repo
        self.fusion_dir = Path(fusion_dir) if fusion_dir else None
        self.name = name
        self._registry = None
        self._fusion = None

    def _load(self):
        if self._registry is not None:
            return
        import_backend()
        from app.core.config import settings
        from app.services.model_registry import ModelRegistry, get_fusion_wrapper_class

        repo = self.fusion_repo or settings.HF_FUSION_REPO_ID
        self.fusion_repo = repo
        reg = ModelRegistry()
        _run_async(reg.load_from_fusion_repo(repo))
        self._registry = reg
        fusion = reg.get_fusion()
        if self.fusion_dir is not None:
            cfg = json.loads((self.fusion_dir / "config.json").read_text(encoding="utf-8"))
            cls = get_fusion_wrapper_class(cfg)
            fusion = cls(repo_id=str(self.fusion_dir), config=cfg, local_path=str(self.fusion_dir))
            fusion.load()
        self._fusion = fusion
        self.threshold = float(fusion.config.get("threshold", 0.5))

    def predict(self, samples, resolve) -> Predictions:
        self._load()
        subs = self._registry.get_all_submodels()
        fusion = self._fusion

        def fn(data: bytes):
            outs = {name: w.predict(image_bytes=data, explain=False) for name, w in subs.items()}
            fused = fusion.predict(submodel_outputs=outs)
            return float(fused["prob_fake"]), {k: float(v["prob_fake"]) for k, v in outs.items()}

        return self._loop(samples, resolve, fn, list(subs.keys()))

    def info(self):
        d = super().info()
        if self._registry is not None:
            d["fusion_repo"] = self.fusion_repo
            d["fusion_config"] = self._fusion.config
            d["submodels"] = {name: {"repo_id": w.repo_id, "config": w.config}
                              for name, w in self._registry.get_all_submodels().items()}
        if self.fusion_dir is not None:
            d["fusion_dir"] = str(self.fusion_dir)
        return d


class BackendSubmodelCandidate(_ImageLoopMixin, Candidate):
    """One backend submodel wrapper, loaded on its own."""

    def __init__(self, key: str, fusion_repo: Optional[str] = None, name: Optional[str] = None):
        self.key = key
        self.fusion_repo = fusion_repo
        self.name = name or f"submodel-{key.split('/')[-1]}"
        self._wrapper = None

    def _load(self):
        if self._wrapper is not None:
            return
        import_backend()
        from app.core.config import settings
        from app.services.hf_hub_service import get_hf_hub_service
        from app.services.model_registry import ModelRegistry, get_wrapper_class

        hf = get_hf_hub_service()
        reg = ModelRegistry()
        repo_id = self.key
        if "/" not in self.key:
            fusion_repo = self.fusion_repo or settings.HF_FUSION_REPO_ID
            cfg = reg._read_config(hf.download_repo(fusion_repo))
            matches = [r for r in _submodel_repo_ids(cfg)
                       if r.split("/")[-1].replace("-final", "") == self.key.replace("-final", "")]
            if not matches:
                raise ValueError(f"submodel {self.key!r} not in {fusion_repo} config: {_submodel_repo_ids(cfg)}")
            repo_id = matches[0]
        path = hf.download_repo(repo_id)
        config = reg._read_config(path)
        wrapper = get_wrapper_class(config)(repo_id=repo_id, config=config, local_path=path)
        wrapper.load()
        self._wrapper = wrapper
        self.threshold = float(config.get("threshold", 0.5))

    def predict(self, samples, resolve) -> Predictions:
        self._load()
        w = self._wrapper

        def fn(data: bytes):
            p = float(w.predict(image_bytes=data, explain=False)["prob_fake"])
            return p, {w.name: p}

        return self._loop(samples, resolve, fn, [w.name])

    def info(self):
        d = super().info()
        if self._wrapper is not None:
            d.update(repo_id=self._wrapper.repo_id, config=self._wrapper.config)
        return d


# ---------------------------------------------------------------------------
# Offline
# ---------------------------------------------------------------------------

class PrecomputedCandidate(Candidate):
    """Scores a column of a probability file, aligned to samples by path."""

    def __init__(self, path: os.PathLike, column: Optional[str] = None, name: Optional[str] = None,
                 threshold: float = 0.5):
        self.path = Path(path)
        self.table: ProbTable = read_probs(self.path)
        col = column or PROB_COL
        if not self.table.has(col) and self.table.has(SUB_PREFIX + col):
            col = SUB_PREFIX + col
        if not self.table.has(col):
            raise ValueError(f"column {col!r} not in {self.path}; have {list(self.table.columns)}")
        self.column = col
        self.name = name or safe_name(f"{self.path}#{column}" if column else str(self.path))
        self.threshold = threshold

    def samples(self) -> List[Sample]:
        return self.table.samples()

    def predict(self, samples, resolve) -> Predictions:
        idx = align_to_samples(self.table, samples)
        vals = self.table.floats(self.column)
        prob = np.where(idx >= 0, vals[np.clip(idx, 0, None)], np.nan) if len(vals) else np.full(len(samples), np.nan)
        subs = {}
        for k in self.table.submodels:
            v = self.table.floats(SUB_PREFIX + k)
            subs[k] = np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan)
        errors = {i: "missing from probability file" for i in np.flatnonzero(idx < 0)}
        for i in np.flatnonzero((idx >= 0) & np.isnan(prob)):
            errors[int(i)] = "NaN probability in file"
        return Predictions(prob, subs, errors)

    def info(self):
        from eval.datasets import file_sha256

        d = super().info()
        d.update(path=str(self.path), column=self.column, sha256=file_sha256(self.path))
        return d


class FusionFileCandidate(Candidate):
    """Applies a local fusion_logreg.pkl to the submodel columns of a probability file."""

    def __init__(self, fusion_dir: os.PathLike, probs_path: os.PathLike, name: Optional[str] = None):
        import joblib

        self.fusion_dir = Path(fusion_dir)
        self.config = json.loads((self.fusion_dir / "config.json").read_text(encoding="utf-8"))
        self.model = joblib.load(self.fusion_dir / "fusion_logreg.pkl")
        self.order = list(self.config["submodel_order"])
        self.threshold = float(self.config.get("threshold", 0.5))
        self.probs_path = Path(probs_path)
        self.table = read_probs(self.probs_path)
        missing = [k for k in self.order if not self.table.has(SUB_PREFIX + k)]
        if missing:
            raise ValueError(f"{self.probs_path} lacks submodel columns {missing}")
        self.name = name or self.fusion_dir.name

    def samples(self) -> List[Sample]:
        return self.table.samples()

    def predict(self, samples, resolve) -> Predictions:
        idx = align_to_samples(self.table, samples)
        X_all = np.column_stack([self.table.floats(SUB_PREFIX + k) for k in self.order])
        prob = np.full(len(samples), np.nan)
        ok = idx >= 0
        X = X_all[idx[ok]]
        finite = np.all(np.isfinite(X), axis=1)
        p = np.full(len(X), np.nan)
        if finite.any():
            p[finite] = self.model.predict_proba(X[finite])[:, 1]
        prob[ok] = p
        subs = {k: np.where(ok, X_all[np.clip(idx, 0, None), j], np.nan) for j, k in enumerate(self.order)}
        errors = {int(i): "missing from probability file" for i in np.flatnonzero(~ok)}
        for i in np.flatnonzero(ok & np.isnan(prob)):
            errors.setdefault(int(i), "missing submodel probability")
        return Predictions(prob, subs, errors)

    def info(self):
        d = super().info()
        d.update(fusion_dir=str(self.fusion_dir), probs=str(self.probs_path), fusion_config=self.config)
        return d


def load_candidate(spec: str, probs: Optional[str] = None, fusion_repo: Optional[str] = None,
                   name: Optional[str] = None, threshold: Optional[float] = None) -> Candidate:
    cand: Candidate
    base, _, column = spec.partition("#")
    p = Path(base)
    if spec == "deployed" or spec.startswith("deployed@"):
        repo = spec.split("@", 1)[1] if "@" in spec else fusion_repo
        cand = BackendStackCandidate(fusion_repo=repo, name=name or "deployed")
    elif spec.startswith("submodel:"):
        key, _, repo = spec[len("submodel:"):].partition("@")
        cand = BackendSubmodelCandidate(key, fusion_repo=repo or fusion_repo, name=name)
    elif p.suffix.lower() in {".csv", ".npz"}:
        cand = PrecomputedCandidate(p, column or None, name=name)
    elif p.is_dir() and (p / "fusion_logreg.pkl").exists():
        if probs:
            cand = FusionFileCandidate(p, probs, name=name)
        else:
            cand = BackendStackCandidate(fusion_repo=fusion_repo, fusion_dir=p, name=name or p.name)
    else:
        raise ValueError(f"Unrecognised candidate spec: {spec!r}")
    if threshold is not None:
        cand.threshold_override = float(threshold)
    return cand
