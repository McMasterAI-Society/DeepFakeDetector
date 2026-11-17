"""
Render a markdown comparison of result JSONs (for PR comments).

    python -m eval.compare eval/results/baseline-2026-11 eval/results/my-candidate --out cmp.md
    python -m eval.compare base/core.json cand/core.json

The first argument is the baseline. Directories are matched by suite file
name. When both results have a ``.preds.csv`` next to them, a paired DeLong
test on the common images is reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from eval import metrics as M
from eval.probfile import read_probs

COLUMNS = [("balanced_accuracy", "Bal. acc"), ("tpr_at_1fpr", "TPR@1%FPR"),
           ("fpr_at_95tpr", "FPR@95%TPR"), ("ece", "ECE"), ("brier", "Brier")]


def load_results(paths: Sequence[str]) -> List[Dict[str, Dict[str, Any]]]:
    """For each path, a mapping suite -> result (with ``_path`` set)."""
    out = []
    for p in map(Path, paths):
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        d = {}
        for f in files:
            r = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(r, dict) or "metrics" not in r:
                continue
            r["_path"] = str(f)
            d[r.get("suite", f.stem)] = r
        out.append(d)
    return out


def _fmt(v: Optional[float], digits: int = 4) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def _delta(v: Optional[float], b: Optional[float], metric: str = "auroc") -> str:
    if v is None or b is None:
        return "–"
    d = v - b
    good = d < 0 if metric in M.LOWER_IS_BETTER else d > 0
    mark = "" if abs(d) < 5e-4 else (" ▲" if good else " ▼")
    return f"{d:+.4f}{mark}"


def _preds(res: Dict[str, Any]) -> Optional[Dict[str, Tuple[int, float]]]:
    name = res.get("preds_file")
    if not name:
        return None
    p = Path(res["_path"]).with_name(name)
    if not p.exists():
        return None
    t = read_probs(p)
    paths, labels = t.strings("path"), t.strings("label")
    variants = t.strings("variant")
    prob = t.floats("prob_fake")
    out = {}
    for i in range(len(t)):
        if variants[i] not in ("", "original") or not np.isfinite(prob[i]):
            continue
        out[paths[i]] = (1 if labels[i].strip().lower() in ("1", "fake") else 0, float(prob[i]))
    return out


def paired_delong(base: Dict[str, Any], cand: Dict[str, Any]) -> Optional[Dict[str, float]]:
    a, b = _preds(base), _preds(cand)
    if not a or not b:
        return None
    keys = sorted(k for k in set(a) & set(b) if a[k][0] == b[k][0])
    if not keys:
        return None
    y = np.array([a[k][0] for k in keys])
    t = M.delong_test(y, np.array([b[k][1] for k in keys]), np.array([a[k][1] for k in keys]))
    if t:
        t["n"] = len(keys)
    return t


def _ci(m: Dict[str, Any], key: str) -> str:
    ci = (m.get("ci95") or {}).get(key)
    return f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""


def render_suite(suite: str, results: List[Tuple[str, Dict[str, Any]]], max_generators: int = 30) -> str:
    base_name, base = results[0]
    bm = base["metrics"]
    lines = [f"### Suite `{suite}`", ""]
    if base.get("suite_description"):
        lines += [f"_{base['suite_description']}_", ""]
    hdr = ["Candidate", "n", "AUROC [95% CI]", "ΔAUROC", "DeLong p"] + [c[1] for c in COLUMNS]
    lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for i, (name, r) in enumerate(results):
        m = r["metrics"]
        if i == 0:
            d, p = "(baseline)", "–"
        else:
            d = _delta(m.get("auroc"), bm.get("auroc"))
            t = paired_delong(base, r)
            p = "–" if t is None else (f"{t['p_value']:.3g}" + ("" if t["n"] == m.get("n") else f" (n={t['n']})"))
        cells = [f"`{name}`", str(r.get("n_scored", m.get("n"))), _fmt(m.get("auroc")) + _ci(m, "auroc"), d, p]
        for key, _ in COLUMNS:
            v = _fmt(m.get(key))
            if i > 0:
                v += f" ({_delta(m.get(key), bm.get(key), key)})"
            cells.append(v)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    gens = sorted({g for _, r in results for g in (r.get("breakdowns", {}).get("generator") or {})})
    if gens:
        gb = base.get("breakdowns", {}).get("generator", {})
        gens = sorted(gens, key=lambda g: -(gb.get(g, {}).get("n_fake") or 0))[:max_generators]
        lines += ["<details><summary>Per-generator AUROC (fakes of that generator vs all reals) "
                  "and TPR at threshold</summary>", ""]
        hdr = ["Generator", "n_fake"] + [f"`{n}` AUROC / TPR" for n, _ in results]
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for g in sorted(gens):
            row = [g, str(gb.get(g, {}).get("n_fake", "–"))]
            for i, (_, r) in enumerate(results):
                e = r.get("breakdowns", {}).get("generator", {}).get(g, {})
                cell = f"{_fmt(e.get('auroc'))} / {_fmt(e.get('tpr'), 3)}"
                if i > 0 and gb.get(g):
                    cell += f" ({_delta(e.get('auroc'), gb[g].get('auroc'))})"
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "</details>", ""]

    for key, title in (("format", "format"), ("size_bucket", "size bucket")):
        vals = sorted({v for _, r in results for v in (r.get("breakdowns", {}).get(key) or {})})
        if not vals:
            continue
        lines += [f"<details><summary>Per-{title} breakdown (AUROC / TNR / TPR)</summary>", ""]
        hdr = [title.capitalize(), "n (real/fake)"] + [f"`{n}`" for n, _ in results]
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        bb = base.get("breakdowns", {}).get(key, {})
        for v in vals:
            e0 = bb.get(v, {})
            row = [v, f"{e0.get('n_real', '–')}/{e0.get('n_fake', '–')}"]
            for _, r in results:
                e = r.get("breakdowns", {}).get(key, {}).get(v, {})
                row.append(f"{_fmt(e.get('auroc'), 3)} / {_fmt(e.get('tnr'), 3)} / {_fmt(e.get('tpr'), 3)}")
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "</details>", ""]

    if any("bias_probe" in r for _, r in results):
        variants = []
        for _, r in results:
            for v in (r.get("bias_probe") or {}).get("variants", {}):
                if v not in variants:
                    variants.append(v)
        lines += ["**Bias probe**: AUROC per variant (drop vs original; DeLong p). "
                  "Large drops mean the model relies on format/size shortcuts.", ""]
        hdr = ["Variant"] + [f"`{n}`" for n, _ in results]
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for v in variants:
            row = [v]
            for _, r in results:
                e = (r.get("bias_probe") or {}).get("variants", {}).get(v)
                if not e:
                    row.append("–")
                    continue
                cell = _fmt(e.get("auroc"))
                if e.get("auroc_drop") is not None:
                    cell += f" (drop {e['auroc_drop']:+.4f}; p={e.get('delong_p', float('nan')):.2g})"
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")
        row = ["**max drop**"] + [_fmt((r.get("bias_probe") or {}).get("max_auroc_drop")) for _, r in results]
        lines += ["| " + " | ".join(row) + " |", ""]

    sc = base.get("metadata_shortcut_auroc") or {}
    if sc:
        parts = ", ".join(f"{k}={_fmt(v, 3)}" for k, v in sc.items())
        lines += [f"Metadata-only AUROC on this suite (how predictable the label is from {', '.join(sc)} "
                  f"alone): {parts}. Values near 1.0 mean this suite cannot tell real detection "
                  "from a format/size shortcut.", ""]
    return "\n".join(lines)


def render(paths: Sequence[str], suites: Optional[Sequence[str]] = None, names: Optional[Sequence[str]] = None) -> str:
    loaded = load_results(paths)
    if names is None:
        names = []
        for p, d in zip(paths, loaded):
            first = next(iter(d.values()), {})
            names.append((first.get("candidate") or {}).get("name") or Path(p).stem)
    all_suites = [s for s in loaded[0]] if loaded else []
    if suites:
        all_suites = [s for s in all_suites if s in suites]
    out = ["## Evaluation comparison", "",
           f"Baseline: `{names[0]}`. ΔAUROC and other deltas are candidate minus baseline "
           "(▲ better, ▼ worse). CIs are stratified bootstrap 95%.", ""]
    for suite in all_suites:
        rows = [(n, d[suite]) for n, d in zip(names, loaded) if suite in d]
        if rows:
            out.append(render_suite(suite, rows))
    return "\n".join(out).rstrip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="result JSONs or result directories; first is the baseline")
    ap.add_argument("--suites", help="comma-separated suites to include")
    ap.add_argument("--names", help="comma-separated display names (default: candidate names)")
    ap.add_argument("--out", help="write markdown here instead of stdout")
    args = ap.parse_args(argv)
    md = render(args.results, suites=args.suites.split(",") if args.suites else None,
                names=args.names.split(",") if args.names else None)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    else:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
