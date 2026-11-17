"""
Score a candidate on an evaluation suite.

    python -m eval.run_eval --candidate deployed --suite core \\
        --manifest eval/manifests/core.csv --data-root /content/data \\
        --out eval/results/deployed/core.json

Writes ``<out>.json`` (metrics, CIs, breakdowns, provenance) and, next to it,
``<suite>.preds.csv`` with per-image probabilities (including per-submodel
``prob.<name>`` columns for in-process candidates). The preds file can be
re-scored offline (``--candidate <file>.preds.csv``), compared with DeLong in
``eval/compare.py``, and used to refit the fusion (``eval/fusion_refit.py``).

``--suite all`` runs every suite whose default manifest exists and writes
``eval/results/<name>/<suite>.json`` for each.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from eval import bias_probe
from eval import metrics as M
from eval.candidates import Candidate, load_candidate, safe_name
from eval.datasets import Manifest, read_manifest
from eval.probfile import write_probs

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "eval" / "manifests"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
SCHEMA = "deepfake-eval/v1"

SUITES: Dict[str, str] = {
    "core": "AI-GenBench benchmark_ten_percent test split.",
    "heldout-gen": "Generators never used to train any submodel or the fusion (FLUX, Midjourney, "
                   "DALL-E 3, Nano Banana, GPT-image, ...). Read per-generator rows.",
    "in-the-wild": "Social-media style images: re-compressed, resized, screenshots.",
    "bias-probe": "Same images with matched formats/sizes (see eval/bias_probe.py). "
                  "Reports AUROC drop vs original.",
}


def default_manifest(suite: str) -> Path:
    for ext in (".csv", ".jsonl"):
        p = MANIFEST_DIR / f"{suite}{ext}"
        if p.exists():
            return p
    return MANIFEST_DIR / f"{suite}.csv"


def _git_info() -> Dict[str, Any]:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def _env_info() -> Dict[str, Any]:
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "sklearn", "torch", "timm", "PIL"):
        m = sys.modules.get(mod)
        if m is not None:
            env[mod] = getattr(m, "__version__", "?")
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            env["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            pass
    return env


def _clean(o):
    """Make an object JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def evaluate(candidate: Candidate, manifest: Manifest, suite: str, n_boot: int = 1000,
             seed: int = 0, preds_path: Optional[Path] = None) -> Dict[str, Any]:
    samples = manifest.samples
    preds = candidate.predict(samples, manifest.resolve)
    threshold = candidate.effective_threshold
    if preds_path is not None:
        write_probs(preds_path, samples, preds.prob_fake, preds.sub_probs, preds.errors)

    p = preds.prob_fake
    ok = np.isfinite(p)
    is_probe = suite == "bias-probe" or any("variant" in s.extra for s in samples[:1])
    if is_probe:
        ok &= np.array([s.extra.get("variant", "original") == "original" for s in samples])
    idx = np.flatnonzero(ok)
    scored = [samples[i] for i in idx]
    y = np.array([s.label for s in scored], dtype=int)
    s = p[idx]
    attrs = {k: [smp.get(k) for smp in scored] for k in ("generator", "source", "format", "size_bucket")}
    rep = M.full_report(y, s, attrs, threshold=threshold, n_boot=n_boot, seed=seed)

    sub_metrics = {}
    for name, arr in preds.sub_probs.items():
        a = arr[idx]
        fin = np.isfinite(a)
        if fin.sum():
            sub_metrics[name] = M.compute_metrics(y[fin], a[fin], 0.5)
            sub_metrics[name]["generator"] = M.breakdown(
                y[fin], a[fin], [attrs["generator"][i] for i in np.flatnonzero(fin)], 0.5,
                fakes_vs_all_reals=True, real_key="real")

    failures = sorted(preds.errors.items())
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "suite": suite,
        "suite_description": SUITES.get(suite, ""),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "candidate": candidate.info(),
        "manifest": {**manifest.info(), "data_root": str(manifest.root) if manifest.root else None},
        "git": _git_info(),
        "env": _env_info(),
        "n_scored": int(len(idx)),
        "n_failed": int(len(failures)),
        "failures_sample": [{"path": samples[i].path, "error": e} for i, e in failures[:20]],
        "seconds": preds.seconds,
        "bootstrap": {"n_boot": n_boot, "seed": seed, "stratified": True},
        "metrics": rep["overall"],
        "breakdowns": rep["breakdowns"],
        "metadata_shortcut_auroc": rep["metadata_shortcut_auroc"],
        "submodels": sub_metrics,
        "preds_file": preds_path.name if preds_path is not None else None,
    }
    if is_probe:
        result["bias_probe"] = bias_probe.report(samples, p, n_boot=n_boot, seed=seed)
    return _clean(result)


def _summary_line(res: Dict[str, Any]) -> str:
    m = res["metrics"]
    f = lambda v: "n/a" if v is None else f"{v:.4f}"
    line = (f"[{res['suite']}] {res['candidate']['name']}: n={res['n_scored']} "
            f"AUROC={f(m['auroc'])} balAcc={f(m['balanced_accuracy'])} "
            f"TPR@1%FPR={f(m['tpr_at_1fpr'])} ECE={f(m['ece'])}")
    if "bias_probe" in res:
        line += f" maxDrop={f(res['bias_probe'].get('max_auroc_drop'))}"
    return line


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.run_eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True,
                    help="deployed[@repo] | submodel:<name>[@repo] | <fusion dir> | <probs.csv|npz>[#col]")
    ap.add_argument("--suite", required=True,
                    help=f"one of {', '.join(SUITES)}, 'all', or any other name (e.g. benchmark-train "
                         "to produce probabilities for eval/fusion_refit.py)")
    ap.add_argument("--manifest", help="manifest CSV/JSONL (default: eval/manifests/<suite>.csv)")
    ap.add_argument("--data-root", default=os.environ.get("EVAL_DATA_ROOT"),
                    help="root that manifest paths are relative to (default: manifest dir, or $EVAL_DATA_ROOT)")
    ap.add_argument("--out", help="result JSON (default: eval/results/<name>/<suite>.json)")
    ap.add_argument("--name", help="candidate name used for the results folder, e.g. baseline-2026-11")
    ap.add_argument("--probs", help="probability file for a local fusion dir candidate (offline fusion)")
    ap.add_argument("--fusion-repo", help="fusion repo for deployed/submodel candidates")
    ap.add_argument("--threshold", type=float, help="override the decision threshold")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="score only the first N rows (smoke tests)")
    ap.add_argument("--no-preds", action="store_true", help="do not write the per-image preds file")
    args = ap.parse_args(argv)

    cand = load_candidate(args.candidate, probs=args.probs, fusion_repo=args.fusion_repo,
                          name=args.name, threshold=args.threshold)
    name = args.name or safe_name(cand.name)
    suites = [s for s in SUITES] if args.suite == "all" else [args.suite]
    if args.suite == "all" and (args.manifest or args.out):
        ap.error("--manifest/--out cannot be combined with --suite all")

    ran = 0
    for suite in suites:
        mpath = Path(args.manifest) if args.manifest else default_manifest(suite)
        if mpath.exists():
            manifest = read_manifest(mpath, root=args.data_root)
        elif hasattr(cand, "samples") and args.manifest is None and args.suite != "all":
            # Precomputed file used as its own manifest.
            manifest = Manifest(cand.samples(), root=None)
        else:
            print(f"[skip] {suite}: manifest {mpath} not found", file=sys.stderr)
            continue
        if args.limit:
            manifest.samples = manifest.samples[: args.limit]
        out = Path(args.out) if args.out else RESULTS_DIR / name / f"{suite}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        preds_path = None if args.no_preds else out.with_name(out.stem + ".preds.csv")
        res = evaluate(cand, manifest, suite, n_boot=args.n_boot, seed=args.seed, preds_path=preds_path)
        out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(_summary_line(res))
        print(f"  -> {out}")
        ran += 1
    return 0 if ran else 1


if __name__ == "__main__":
    raise SystemExit(main())
