"""
Detection metrics, bootstrap confidence intervals, DeLong's paired AUROC test,
and per-group breakdowns.

Convention: label 1 = fake (positive class), scores are P(fake).
Metrics that are undefined for a subset (e.g. AUROC with a single class)
are reported as ``None``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

CORE_METRICS = ["auroc", "balanced_accuracy", "tpr_at_1fpr", "fpr_at_95tpr", "ece", "brier"]
LOWER_IS_BETTER = {"fpr_at_95tpr", "ece", "brier"}


def _arrays(y, s):
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(s, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"labels and scores differ in shape: {y.shape} vs {s.shape}")
    return y, s


def _both_classes(y) -> bool:
    return bool(np.any(y == 1) and np.any(y == 0))


def midrank(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, starts, counts = np.unique(xs, return_index=True, return_counts=True)
    avg = starts + (counts + 1) / 2.0
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.repeat(avg, counts)
    return ranks


def auroc(y, s) -> Optional[float]:
    """Area under the ROC curve via the Mann-Whitney statistic (ties count 1/2)."""
    y, s = _arrays(y, s)
    if not _both_classes(y):
        return None
    m = int((y == 1).sum())
    n = len(y) - m
    r = midrank(s)
    return float((r[y == 1].sum() - m * (m + 1) / 2.0) / (m * n))


def roc_points(y, s):
    """Full ROC curve (no intermediate points dropped), starting at (0, 0).

    Returns (fpr, tpr, thresholds) with thresholds decreasing; a sample is
    predicted fake when score >= threshold.
    """
    y, s = _arrays(y, s)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, len(y) - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = (1 + idx) - tps
    P, N = max(int(y.sum()), 1), max(int(len(y) - y.sum()), 1)
    tpr = np.r_[0.0, tps / P]
    fpr = np.r_[0.0, fps / N]
    thr = np.r_[np.inf, s_sorted[idx]]
    return fpr, tpr, thr


def tpr_at_fpr(y, s, target_fpr: float = 0.01) -> Optional[float]:
    """Highest TPR reachable with FPR <= target (no interpolation)."""
    y, s = _arrays(y, s)
    if not _both_classes(y):
        return None
    fpr, tpr, _ = roc_points(y, s)
    return float(tpr[fpr <= target_fpr + 1e-12].max())


def fpr_at_tpr(y, s, target_tpr: float = 0.95) -> Optional[float]:
    """Lowest FPR reachable with TPR >= target (no interpolation)."""
    y, s = _arrays(y, s)
    if not _both_classes(y):
        return None
    fpr, tpr, _ = roc_points(y, s)
    return float(fpr[tpr >= target_tpr - 1e-12].min())


def balanced_accuracy(y, s, threshold: float = 0.5) -> Optional[float]:
    y, s = _arrays(y, s)
    if not _both_classes(y):
        return None
    pred = (s >= threshold).astype(int)
    tpr = float((pred[y == 1] == 1).mean())
    tnr = float((pred[y == 0] == 0).mean())
    return (tpr + tnr) / 2.0


def ece(y, s, n_bins: int = 15) -> Optional[float]:
    """Expected calibration error of P(fake) with equal-width bins.

    Matches ``compute_ece`` in the notebooks: the first bin is closed on both
    ends, the rest are (lo, hi].
    """
    y, s = _arrays(y, s)
    if len(y) == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index: 0 for s in [0, e1], i for s in (e_i, e_{i+1}]
    idx = np.clip(np.searchsorted(edges, s, side="left") - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        k = int(mask.sum())
        if k:
            total += (k / len(s)) * abs(float(y[mask].mean()) - float(s[mask].mean()))
    return float(total)


def brier(y, s) -> Optional[float]:
    y, s = _arrays(y, s)
    if len(y) == 0:
        return None
    return float(np.mean((s - y) ** 2))


def compute_metrics(y, s, threshold: float = 0.5, n_bins: int = 15) -> Dict[str, Any]:
    y, s = _arrays(y, s)
    out = {
        "n": int(len(y)),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "threshold": float(threshold),
        "auroc": auroc(y, s),
        "balanced_accuracy": balanced_accuracy(y, s, threshold),
        "tpr_at_1fpr": tpr_at_fpr(y, s, 0.01),
        "fpr_at_95tpr": fpr_at_tpr(y, s, 0.95),
        "ece": ece(y, s, n_bins),
        "brier": brier(y, s),
    }
    if out["n_fake"]:
        out["tpr"] = float((s[y == 1] >= threshold).mean())
    if out["n_real"]:
        out["tnr"] = float((s[y == 0] < threshold).mean())
    return out


def bootstrap_ci(y, s, threshold: float = 0.5, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0, metrics: Sequence[str] = CORE_METRICS,
                 n_bins: int = 15) -> Dict[str, Optional[List[float]]]:
    """Percentile bootstrap CIs, resampling reals and fakes separately (stratified)."""
    y, s = _arrays(y, s)
    if n_boot <= 0 or not _both_classes(y):
        return {m: None for m in metrics}
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    fns: Dict[str, Callable] = {
        "auroc": auroc,
        "balanced_accuracy": lambda a, b: balanced_accuracy(a, b, threshold),
        "tpr_at_1fpr": lambda a, b: tpr_at_fpr(a, b, 0.01),
        "fpr_at_95tpr": lambda a, b: fpr_at_tpr(a, b, 0.95),
        "ece": lambda a, b: ece(a, b, n_bins),
        "brier": brier,
    }
    vals = {m: np.empty(n_boot) for m in metrics}
    for i in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        yb, sb = y[idx], s[idx]
        for m in metrics:
            vals[m][i] = fns[m](yb, sb)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {m: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for m, v in vals.items()}


# ---------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------

def _delong_components(y, scores: np.ndarray):
    """Fast DeLong (Sun & Xu, 2014). scores: (k, n). Returns aucs (k,), cov (k, k)."""
    pos = scores[:, y == 1]
    neg = scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.vstack([midrank(pos[i]) for i in range(k)])
    ty = np.vstack([midrank(neg[i]) for i in range(k)])
    tz = np.vstack([midrank(np.r_[pos[i], neg[i]]) for i in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01)) if m > 1 else np.zeros((k, k))
    sy = np.atleast_2d(np.cov(v10)) if n > 1 else np.zeros((k, k))
    return aucs, sx / m + sy / n


def _two_sided_p(z: float) -> float:
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def delong_ci(y, s, alpha: float = 0.05) -> Optional[Dict[str, float]]:
    """AUROC with a DeLong normal-approximation CI."""
    y, s = _arrays(y, s)
    if not _both_classes(y):
        return None
    aucs, cov = _delong_components(y, s[None, :])
    se = math.sqrt(max(float(cov[0, 0]), 0.0))
    z = _z(1 - alpha / 2)
    return {"auroc": float(aucs[0]), "se": se,
            "ci": [max(0.0, aucs[0] - z * se), min(1.0, aucs[0] + z * se)]}


def _z(q: float) -> float:
    # Inverse normal CDF via bisection on erfc; only used for CI widths.
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def delong_test(y, s1, s2) -> Optional[Dict[str, float]]:
    """Paired two-sided DeLong test for AUROC(s1) vs AUROC(s2) on the same samples."""
    y, s1 = _arrays(y, s1)
    _, s2 = _arrays(y, s2)
    if not _both_classes(y):
        return None
    aucs, cov = _delong_components(y, np.vstack([s1, s2]))
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    diff = float(aucs[0] - aucs[1])
    if var <= 1e-15:
        z, p = (0.0, 1.0) if abs(diff) < 1e-12 else (math.copysign(math.inf, diff), 0.0)
    else:
        z = diff / math.sqrt(var)
        p = _two_sided_p(z)
    return {"auroc_1": float(aucs[0]), "auroc_2": float(aucs[1]), "diff": diff,
            "se": math.sqrt(max(var, 0.0)), "z": float(z), "p_value": float(p)}


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------

def _group_entry(y, s, threshold) -> Dict[str, Any]:
    e: Dict[str, Any] = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
                         "mean_prob_fake": float(s.mean()) if len(s) else None}
    e["auroc"] = auroc(y, s)
    e["balanced_accuracy"] = balanced_accuracy(y, s, threshold)
    if e["n_fake"]:
        e["tpr"] = float((s[y == 1] >= threshold).mean())
    if e["n_real"]:
        e["tnr"] = float((s[y == 0] < threshold).mean())
    return e


def breakdown(y, s, keys: Sequence[str], threshold: float = 0.5,
              fakes_vs_all_reals: bool = False, real_key: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Metrics per value of ``keys``.

    With ``fakes_vs_all_reals`` (used for generators), each group contains the
    fakes of that value plus *all* reals, so AUROC answers "how well are this
    generator's fakes separated from real images". Real rows (whose key is
    ``real_key``) do not form their own group in that mode.
    """
    y, s = _arrays(y, s)
    keys = np.asarray([str(k) for k in keys])
    out: Dict[str, Dict[str, Any]] = {}
    if fakes_vs_all_reals:
        real_mask = y == 0
        for v in sorted(set(keys[y == 1])):
            if real_key is not None and v == real_key:
                continue
            mask = real_mask | ((y == 1) & (keys == v))
            out[v] = _group_entry(y[mask], s[mask], threshold)
    else:
        for v in sorted(set(keys)):
            mask = keys == v
            out[v] = _group_entry(y[mask], s[mask], threshold)
    return out


def metadata_shortcut_auroc(y, keys: Sequence[str]) -> Optional[float]:
    """In-sample AUROC of a classifier that only sees a categorical metadata field.

    Each value is scored by its empirical fake rate. Near 1.0 means the label is
    almost determined by e.g. file format, so a detector can score well by
    learning that shortcut instead of generation artifacts.
    """
    y = np.asarray(y).astype(int)
    keys = np.asarray([str(k) for k in keys])
    if not _both_classes(y):
        return None
    rate: Dict[str, float] = {}
    for v in set(keys):
        rate[v] = float(y[keys == v].mean())
    return auroc(y, np.array([rate[k] for k in keys]))


def full_report(y, s, attrs: Dict[str, Sequence[str]], threshold: float = 0.5, n_boot: int = 1000,
                seed: int = 0) -> Dict[str, Any]:
    """Overall metrics + CIs + breakdowns + shortcut diagnostics.

    ``attrs`` maps attribute name (generator, source, format, size_bucket) to
    per-sample values.
    """
    y, s = _arrays(y, s)
    overall = compute_metrics(y, s, threshold)
    overall["ci95"] = bootstrap_ci(y, s, threshold, n_boot=n_boot, seed=seed)
    overall["auroc_delong"] = delong_ci(y, s)
    breakdowns: Dict[str, Any] = {}
    if "generator" in attrs:
        breakdowns["generator"] = breakdown(y, s, attrs["generator"], threshold,
                                            fakes_vs_all_reals=True, real_key="real")
    for k in ("source", "format", "size_bucket"):
        if k in attrs:
            breakdowns[k] = breakdown(y, s, attrs[k], threshold)
    shortcut = {}
    for k in ("format", "size_bucket"):
        if k in attrs:
            shortcut[k] = metadata_shortcut_auroc(y, attrs[k])
    if "format" in attrs and "size_bucket" in attrs:
        combo = [f"{a}|{b}" for a, b in zip(attrs["format"], attrs["size_bucket"])]
        shortcut["format+size_bucket"] = metadata_shortcut_auroc(y, combo)
    return {"overall": overall, "breakdowns": breakdowns, "metadata_shortcut_auroc": shortcut}
