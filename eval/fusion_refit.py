"""
Refit the logistic-regression fusion from probability files.

Why: the deployed ``fusion-logreg-final`` was re-fit from scratch on ~840
manual images only (fakes = generator PNGs, reals = OpenImages JPEGs, a likely
format shortcut) with a random split that leaks ``prompt_id`` across train and
test. This tool refits on any combination of probability files (e.g.
benchmark train + manual train, together rather than sequentially) with:

* logit features ``log(p / (1 - p))`` (clipped) instead of raw probabilities;
* class-balanced sample weights (optionally also balancing datasets);
* GROUPED splits: StratifiedGroupKFold keyed by ``prompt_id`` (fallback: path),
  so images from the same prompt never sit on both sides of a split;
* C selected by leave-one-generator-out (LOGO) AUROC when there are >= 2 fake
  generators, otherwise by grouped-CV AUROC;
* temperature scaling fit on out-of-fold (grouped) logits, folded into the
  exported coefficients;
* LOGO evaluation report and optional held-out test files.

The export is LOCAL ONLY (default ``eval/.artifacts/fusion-logreg-refit/``, gitignored):
``fusion_logreg.pkl`` + ``config.json`` + ``label_map.json`` + ``refit_report.json``.
Nothing is uploaded. The pickle is an sklearn Pipeline (clip -> logit -> LR)
that takes raw submodel probabilities, so the current backend
``LogRegFusionWrapper`` (``predict_proba`` on stacked probs) works unchanged.
Pickle with the backend's scikit-learn version (1.6.1) to avoid version skew.

    python -m eval.fusion_refit \\
        --train eval/results/deployed/benchmark-train.preds.csv eval/results/deployed/manual-train.preds.csv \\
        --test  eval/results/deployed/core.preds.csv \\
        --base-config path/to/fusion-logreg-final/config.json \\
        --out-dir eval/.artifacts/fusion-logreg-refit
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from eval import metrics as M
from eval.datasets import file_sha256
from eval.probfile import SUB_PREFIX, read_probs, write_probs

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "eval" / ".artifacts" / "fusion-logreg-refit"
DEFAULT_ORDER = ["cnn-transfer", "vit-base", "deit-distilled", "gradfield-cnn"]
BACKEND_SKLEARN = "1.6.1"
EPS = 1e-6


@dataclass
class FitData:
    X: np.ndarray            # (n, k) raw submodel P(fake)
    y: np.ndarray            # (n,)
    groups: np.ndarray       # (n,) str
    generators: np.ndarray   # (n,) str
    datasets: np.ndarray     # (n,) str: which file the row came from
    formats: np.ndarray      # (n,) str
    samples: list            # eval.datasets.Sample per row
    submodels: List[str]
    files: List[Dict[str, Any]]


def logit(p, eps: float = EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def group_keys(samples, datasets, group_by: str = "prompt_id", fallback: str = "path") -> np.ndarray:
    """Group id per sample.

    Uses ``group_by`` (prompt_id, generator, source...) when non-empty, else
    ``fallback``. Reals never group by generator (that would put every real in
    one group). Keys are prefixed with the dataset so ids from different files
    do not collide.
    """
    out = []
    for s, ds in zip(samples, datasets):
        v = s.get(group_by) if group_by else ""
        if group_by == "generator" and s.label == 0:
            v = ""
        if not v:
            v = s.get(fallback) or s.path
            out.append(f"{ds}:{fallback}:{v}")
        else:
            out.append(f"{ds}:{group_by}:{v}")
    return np.array(out)


def load_fit_data(paths: Sequence[str], submodels: Optional[Sequence[str]] = None,
                  group_by: str = "prompt_id", fallback: str = "path") -> FitData:
    Xs, samples, dsets, files = [], [], [], []
    order: Optional[List[str]] = list(submodels) if submodels else None
    for path in paths:
        t = read_probs(path)
        if order is None:
            have = t.submodels
            order = [m for m in DEFAULT_ORDER if m in have] + [m for m in have if m not in DEFAULT_ORDER]
        missing = [m for m in order if not t.has(SUB_PREFIX + m)]
        if missing:
            raise ValueError(f"{path}: missing submodel columns {missing} (have {t.submodels})")
        X = np.column_stack([t.floats(SUB_PREFIX + m) for m in order])
        err = np.array([bool(e) for e in t.strings("error")])
        keep = np.all(np.isfinite(X), axis=1) & ~err
        smp = [s for s, k in zip(t.samples(), keep) if k]
        name = Path(path).name
        Xs.append(X[keep])
        samples += smp
        dsets += [name] * len(smp)
        files.append({"path": str(path), "sha256": file_sha256(path), "n_rows": len(t),
                      "n_used": int(keep.sum()), "n_fake": int(sum(s.label for s in smp))})
    if not order:
        raise ValueError("no submodel columns (prob.<name>) found")
    datasets = np.array(dsets)
    return FitData(
        X=np.vstack(Xs) if Xs else np.zeros((0, len(order))),
        y=np.array([s.label for s in samples], dtype=int),
        groups=group_keys(samples, datasets, group_by, fallback),
        generators=np.array([s.generator for s in samples]),
        datasets=datasets,
        formats=np.array([s.format for s in samples]),
        samples=samples, submodels=list(order), files=files,
    )


def subset(d: FitData, mask: np.ndarray) -> FitData:
    idx = np.flatnonzero(mask)
    return FitData(d.X[idx], d.y[idx], d.groups[idx], d.generators[idx], d.datasets[idx],
                   d.formats[idx], [d.samples[i] for i in idx], d.submodels, d.files)


def sample_weights(y: np.ndarray, datasets: Optional[np.ndarray] = None,
                   class_weight: str = "balanced", balance_datasets: bool = False) -> np.ndarray:
    w = np.ones(len(y), dtype=float)
    if balance_datasets and datasets is not None:
        names = np.unique(datasets)
        for ds in names:
            m = datasets == ds
            w[m] *= len(y) / (len(names) * m.sum())
    if class_weight == "balanced":
        for c in (0, 1):
            m = y == c
            if m.any():
                w[m] *= w.sum() / (2.0 * w[m].sum())
    return w


def make_pipeline(C: float, features: str = "logit", eps: float = EPS, seed: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    import scipy.special

    steps = []
    if features == "logit":
        # numpy/scipy callables only, so the pickle loads without this package.
        steps.append(("clip", FunctionTransformer(np.clip, kw_args={"a_min": eps, "a_max": 1 - eps})))
        steps.append(("logit", FunctionTransformer(scipy.special.logit)))
    elif features != "prob":
        raise ValueError(f"unknown features {features!r}")
    steps.append(("lr", LogisticRegression(C=C, max_iter=5000, random_state=seed)))
    return Pipeline(steps)


def fit(d: FitData, C: float, features: str, class_weight: str, balance_datasets: bool, seed: int = 42):
    pipe = make_pipeline(C, features, seed=seed)
    w = sample_weights(d.y, d.datasets, class_weight, balance_datasets)
    pipe.fit(d.X, d.y, lr__sample_weight=w)
    return pipe


def grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Fold id per row from StratifiedGroupKFold; no group spans two folds."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_groups = len(np.unique(groups))
    k = max(2, min(n_splits, n_groups))
    fold = np.full(len(y), -1, dtype=int)
    for f, (_, te) in enumerate(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
                                .split(np.zeros(len(y)), y, groups)):
        fold[te] = f
    return fold


def oof_decision(d: FitData, folds: np.ndarray, **fit_kw) -> np.ndarray:
    z = np.full(len(d.y), np.nan)
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        if len(np.unique(d.y[tr])) < 2:
            continue
        z[te] = fit(subset(d, tr), **fit_kw).decision_function(d.X[te])
    return z


def logo_splits(d: FitData, n_folds: int = 5, seed: int = 42):
    """Yield (generator, fold, train_mask, test_mask) for leave-one-generator-out.

    Rows are first assigned to grouped folds (by prompt_id / fallback group).
    For held-out generator g and fold k:

    * test  = fakes of g in fold k + reals in fold k
    * train = rows NOT of generator g and NOT in fold k

    So the held-out generator is never seen, and no prompt group appears on
    both sides (the same prompt rendered by another generator stays out of
    train). Pooling the K test folds covers every fake of g and every real.
    """
    folds = grouped_folds(d.y, d.groups, n_folds, seed)
    fake_gens = sorted(set(d.generators[d.y == 1]))
    for g in fake_gens:
        is_g = (d.y == 1) & (d.generators == g)
        for k in np.unique(folds):
            in_k = folds == k
            test = in_k & (is_g | (d.y == 0))
            train = ~in_k & ~is_g
            yield g, int(k), train, test


def logo_eval(d: FitData, n_folds: int = 5, **fit_kw) -> Dict[str, Any]:
    """Per held-out generator AUROC (fakes of g vs reals), pooled over grouped folds."""
    per: Dict[str, Any] = {}
    pooled: Dict[str, Dict[str, list]] = {}
    for g, k, tr, te in logo_splits(d, n_folds=n_folds, seed=fit_kw.get("seed", 42)):
        if len(np.unique(d.y[tr])) < 2 or not te.any():
            continue
        p = fit(subset(d, tr), **fit_kw).predict_proba(d.X[te])[:, 1]
        acc = pooled.setdefault(g, {"y": [], "p": [], "n_train": []})
        acc["y"].append(d.y[te]); acc["p"].append(p); acc["n_train"].append(int(tr.sum()))
    for g, acc in pooled.items():
        yt, p = np.concatenate(acc["y"]), np.concatenate(acc["p"])
        per[g] = {"auroc": M.auroc(yt, p),
                  "tpr": float((p[yt == 1] >= 0.5).mean()) if (yt == 1).any() else None,
                  "tnr": float((p[yt == 0] < 0.5).mean()) if (yt == 0).any() else None,
                  "n_fake": int(yt.sum()), "n_real": int((yt == 0).sum()),
                  "mean_n_train": float(np.mean(acc["n_train"]))}
    aucs = [v["auroc"] for v in per.values() if v["auroc"] is not None]
    return {"per_generator": per, "mean_auroc": float(np.mean(aucs)) if aucs else None,
            "min_auroc": float(np.min(aucs)) if aucs else None}


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Scalar T minimising NLL of sigmoid(z / T)."""
    from scipy.optimize import minimize_scalar

    ok = np.isfinite(z)
    z, y = z[ok], y[ok]

    def nll(log_t):
        t = np.exp(log_t)
        q = z / t
        return float(np.mean(np.logaddexp(0, q) - y * q))

    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def apply_temperature(pipe, T: float):
    lr = pipe.named_steps["lr"]
    lr.coef_ = lr.coef_ / T
    lr.intercept_ = lr.intercept_ / T
    return pipe


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def evaluate_split(d: FitData, prob: np.ndarray, threshold: float = 0.5, n_boot: int = 1000,
                   seed: int = 0) -> Dict[str, Any]:
    attrs = {"generator": list(d.generators), "format": list(d.formats),
             "source": [s.source for s in d.samples], "size_bucket": [s.size_bucket for s in d.samples]}
    rep = M.full_report(d.y, prob, attrs, threshold=threshold, n_boot=n_boot, seed=seed)
    rep["single_submodel_auroc"] = {m: M.auroc(d.y, d.X[:, j]) for j, m in enumerate(d.submodels)}
    return rep


def _check_no_leak(groups: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
    shared = set(groups[a]) & set(groups[b])
    if shared:
        raise AssertionError(f"group leakage across split: {sorted(shared)[:5]}")


def refit(train_files: Sequence[str], test_files: Sequence[str] = (), out_dir: Path = DEFAULT_OUT,
          submodels: Optional[Sequence[str]] = None, features: str = "logit",
          class_weight: str = "balanced", balance_datasets: bool = False,
          C_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0), group_by: str = "prompt_id",
          fallback_group: str = "path", n_splits: int = 5, select_by: str = "auto",
          temperature: bool = True, holdout: float = 0.0, base_config: Optional[Path] = None,
          submodel_repos: Optional[Sequence[str]] = None, name: str = "fusion-logreg-refit",
          version: Optional[str] = None, reference_fusion: Optional[Path] = None,
          n_boot: int = 1000, seed: int = 42) -> Dict[str, Any]:
    import joblib
    import sklearn

    if sklearn.__version__ != BACKEND_SKLEARN:
        warnings.warn(f"scikit-learn {sklearn.__version__} != backend {BACKEND_SKLEARN}; "
                      "the pickle may not load cleanly in production. Use eval/requirements.txt.")
    data = load_fit_data(train_files, submodels, group_by, fallback_group)
    test = load_fit_data(test_files, data.submodels, group_by, fallback_group) if test_files else None
    if test is None and holdout > 0:
        folds = grouped_folds(data.y, data.groups, int(round(1 / holdout)), seed)
        te = folds == 0
        _check_no_leak(data.groups, te, ~te)
        test, data = subset(data, te), subset(data, ~te)
        test.files = [{"note": f"grouped holdout {holdout:.0%} of train files"}]
    if len(np.unique(data.y)) < 2:
        raise ValueError("training data needs both real and fake rows")

    fit_kw = dict(features=features, class_weight=class_weight, balance_datasets=balance_datasets, seed=seed)
    folds = grouped_folds(data.y, data.groups, n_splits, seed)
    n_fake_gens = len(set(data.generators[data.y == 1]))
    if select_by == "auto":
        select_by = "logo" if n_fake_gens >= 2 else "cv"

    grid = []
    for C in C_grid:
        kw = dict(fit_kw, C=C)
        z = oof_decision(data, folds, **kw)
        ok = np.isfinite(z)
        entry = {"C": C, "cv_oof_auroc": M.auroc(data.y[ok], z[ok])}
        if n_fake_gens >= 2:
            entry["logo"] = logo_eval(data, n_folds=n_splits, **kw)
            entry["logo_mean_auroc"] = entry["logo"]["mean_auroc"]
        grid.append(entry)
    key = "logo_mean_auroc" if select_by == "logo" else "cv_oof_auroc"
    best = max(grid, key=lambda e: (e.get(key) or -1, e.get("cv_oof_auroc") or -1))
    C = best["C"]

    z_oof = oof_decision(data, folds, **dict(fit_kw, C=C))
    T = fit_temperature(z_oof, data.y) if temperature else 1.0
    ok = np.isfinite(z_oof)
    oof_raw = _sigmoid(z_oof[ok])
    oof_cal = _sigmoid(z_oof[ok] / T)
    calibration = {"temperature": T, "fit_on": "grouped out-of-fold logits",
                   "oof_ece_before": M.ece(data.y[ok], oof_raw), "oof_ece_after": M.ece(data.y[ok], oof_cal),
                   "oof_brier_before": M.brier(data.y[ok], oof_raw), "oof_brier_after": M.brier(data.y[ok], oof_cal)}

    pipe = apply_temperature(fit(data, C=C, **fit_kw), T)
    lr = pipe.named_steps["lr"]
    coef = [float(c) for c in lr.coef_[0]]
    intercept = float(lr.intercept_[0])

    report: Dict[str, Any] = {
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "train_files": data.files,
        "n_train": int(len(data.y)), "n_train_fake": int(data.y.sum()),
        "submodel_order": data.submodels,
        "settings": {"features": features, "class_weight": class_weight, "balance_datasets": balance_datasets,
                     "group_by": group_by, "fallback_group": fallback_group, "n_splits": int(len(np.unique(folds))),
                     "select_by": select_by, "C_grid": list(C_grid), "seed": seed},
        "train_metadata_shortcut_auroc": {"format": M.metadata_shortcut_auroc(data.y, data.formats)},
        "grid": grid, "selected_C": C, "calibration": calibration,
        "coef": dict(zip(data.submodels, coef)), "intercept": intercept,
        "cv_oof": M.compute_metrics(data.y[ok], oof_cal),
    }
    if (report["train_metadata_shortcut_auroc"]["format"] or 0) > 0.9:
        report.setdefault("warnings", []).append(
            "Training labels are >0.9 AUROC-predictable from image format alone; "
            "submodel probabilities may encode a format shortcut. Check the bias-probe suite.")
    if test is not None and len(test.y):
        p_test = pipe.predict_proba(test.X)[:, 1]
        report["test_files"] = test.files
        report["test"] = evaluate_split(test, p_test, n_boot=n_boot, seed=seed)
        if reference_fusion is not None:
            ref = joblib.load(Path(reference_fusion) / "fusion_logreg.pkl")
            ref_cfg = json.loads((Path(reference_fusion) / "config.json").read_text(encoding="utf-8"))
            Xr = np.column_stack([test.X[:, data.submodels.index(m)] for m in ref_cfg["submodel_order"]])
            p_ref = ref.predict_proba(Xr)[:, 1]
            report["test_reference"] = {"fusion_dir": str(reference_fusion),
                                        "metrics": M.compute_metrics(test.y, p_ref),
                                        "delong_refit_vs_reference": M.delong_test(test.y, p_test, p_ref)}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "fusion_logreg.pkl")
    if test is not None and len(test.y):
        write_probs(out_dir / "test.preds.csv", test.samples, pipe.predict_proba(test.X)[:, 1],
                    {m: test.X[:, j] for j, m in enumerate(data.submodels)})

    cfg: Dict[str, Any] = {}
    if base_config is not None:
        base = json.loads(Path(base_config).read_text(encoding="utf-8"))
        cfg.update({k: base[k] for k in ("submodels", "labels") if k in base})
        base_order = base.get("submodel_order")
        if base_order and list(base_order) != data.submodels:
            raise ValueError(f"submodel order {data.submodels} != base config {base_order}")
    if submodel_repos:
        cfg["submodels"] = list(submodel_repos)
    if "submodels" not in cfg:
        cfg["submodels"] = [f"DeepFakeDetector/{m}-final" for m in data.submodels]
        report.setdefault("warnings", []).append(
            "config.submodels guessed as DeepFakeDetector/<name>-final; pass --base-config to copy "
            "the deployed list (and pinned revisions) instead.")
    test_auroc = (report.get("test") or {}).get("overall", {}).get("auroc")
    cfg.update({
        "type": "probability_stacking_fusion",
        "name": name,
        "version": version or _dt.date.today().strftime("refit-%Y%m%d"),
        "submodel_order": data.submodels,
        "num_submodels": len(data.submodels),
        "threshold": 0.5,
        "labels": cfg.get("labels", {"0": "real", "1": "fake"}),
        "feature_transform": features,
        "logit_clip_eps": EPS if features == "logit" else None,
        "temperature": T,
        "coef": coef,
        "intercept": intercept,
        "C": C,
        "sklearn_version": sklearn.__version__,
        "test_auroc": test_auroc,
        "metrics_provenance": "computed by eval/fusion_refit.py for exactly these weights; "
                              "see refit_report.json",
    })
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (out_dir / "label_map.json").write_text(json.dumps({"0": "real", "1": "fake"}) + "\n", encoding="utf-8")
    from eval.run_eval import _clean
    (out_dir / "refit_report.json").write_text(json.dumps(_clean(report), indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.fusion_refit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, help="probability files to fit on (combined)")
    ap.add_argument("--test", nargs="*", default=[], help="held-out probability files to report on")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--submodels", help="comma-separated submodel order (default: from prob.* columns)")
    ap.add_argument("--features", choices=["logit", "prob"], default="logit")
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--balance-datasets", action="store_true", help="give each train file equal total weight")
    ap.add_argument("--C-grid", default="0.01,0.1,1,10")
    ap.add_argument("--group-by", default="prompt_id", help="prompt_id | generator | source | ...")
    ap.add_argument("--fallback-group", default="path")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--select-by", choices=["auto", "logo", "cv"], default="auto")
    ap.add_argument("--no-temperature", action="store_true")
    ap.add_argument("--holdout", type=float, default=None,
                    help="grouped holdout fraction of train when no --test (default 0.2 if no --test)")
    ap.add_argument("--base-config", help="deployed fusion config.json to copy submodels/labels from")
    ap.add_argument("--submodel-repos", help="comma-separated repo ids for config.submodels")
    ap.add_argument("--reference-fusion", help="fusion dir (pkl+config) to compare against on --test")
    ap.add_argument("--name", default="fusion-logreg-refit")
    ap.add_argument("--version")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    holdout = a.holdout if a.holdout is not None else (0.0 if a.test else 0.2)
    rep = refit(
        a.train, a.test, Path(a.out_dir),
        submodels=a.submodels.split(",") if a.submodels else None, features=a.features,
        class_weight=a.class_weight, balance_datasets=a.balance_datasets,
        C_grid=[float(c) for c in a.C_grid.split(",")], group_by=a.group_by,
        fallback_group=a.fallback_group, n_splits=a.n_splits, select_by=a.select_by,
        temperature=not a.no_temperature, holdout=holdout,
        base_config=Path(a.base_config) if a.base_config else None,
        submodel_repos=a.submodel_repos.split(",") if a.submodel_repos else None,
        name=a.name, version=a.version,
        reference_fusion=Path(a.reference_fusion) if a.reference_fusion else None,
        n_boot=a.n_boot, seed=a.seed,
    )
    summary = {
        "out_dir": a.out_dir, "selected_C": rep["selected_C"], "temperature": rep["calibration"]["temperature"],
        "coef": rep["coef"], "cv_oof_auroc": rep["cv_oof"]["auroc"],
        "logo_mean_auroc": next((g.get("logo_mean_auroc") for g in rep["grid"] if g["C"] == rep["selected_C"]), None),
        "test_auroc": (rep.get("test") or {}).get("overall", {}).get("auroc"),
        "warnings": rep.get("warnings", []),
    }
    print(json.dumps(summary, indent=2, default=str))
    print("Exported locally only; nothing was uploaded.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
