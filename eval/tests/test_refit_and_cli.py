import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eval import compare, fusion_refit as FR, run_eval
from eval.candidates import FusionFileCandidate, PrecomputedCandidate, load_candidate
from eval.datasets import Sample, write_manifest
from eval.probfile import read_probs, write_probs

SUBS = ["cnn-transfer", "vit-base", "deit-distilled", "gradfield-cnn"]


def synth_probs(path: Path, seed=0, n_prompts=40, gens=("dalle", "nano-banana", "bing"), n_real=120,
                fmt_shortcut=False):
    """Manual-gen-like probability file: every prompt rendered by every generator."""
    rng = np.random.default_rng(seed)
    samples, rows = [], []
    for g_i, g in enumerate(gens):
        for p in range(n_prompts):
            samples.append(Sample(f"images/{g}/p{p:03d}.png", 1, g, "manual-gen", "png", 1024, 1024, f"p{p:03d}"))
    for r in range(n_real):
        samples.append(Sample(f"reals/{r}.jpg", 0, "real", "openimages",
                              "jpeg" if fmt_shortcut or r % 2 else "png", 800, 600, ""))
    y = np.array([s.label for s in samples])
    strength = np.array([2.5, 0.3, 1.0, 0.8])
    z = y[:, None] * strength[None, :] + rng.normal(size=(len(y), 4)) - strength / 2
    X = 1 / (1 + np.exp(-z))
    fused = X.mean(axis=1)
    write_probs(path, samples, fused, {k: X[:, j] for j, k in enumerate(SUBS)})
    return samples, y, X


def test_grouped_folds_have_no_leakage(tmp_path):
    synth_probs(tmp_path / "m.csv")
    d = FR.load_fit_data([tmp_path / "m.csv"])
    folds = FR.grouped_folds(d.y, d.groups, 5, seed=0)
    assert (folds >= 0).all()
    for f in np.unique(folds):
        a, b = set(d.groups[folds == f]), set(d.groups[folds != f])
        assert not (a & b)
        assert set(d.y[folds == f]) == {0, 1}
    # every prompt group holds all 3 generators' renderings
    pid_groups = [g for g in set(d.groups) if ":prompt_id:" in g]
    assert len(pid_groups) == 40
    assert all((d.groups == g).sum() == 3 for g in pid_groups)


def test_logo_splits_hold_out_generator_and_prompts(tmp_path):
    synth_probs(tmp_path / "m.csv")
    d = FR.load_fit_data([tmp_path / "m.csv"])
    seen = set()
    covered = {}
    for g, k, tr, te in FR.logo_splits(d):
        seen.add(g)
        assert not (set(d.groups[tr]) & set(d.groups[te]))
        assert set(d.generators[te & (d.y == 1)]) <= {g}
        assert not ((d.generators == g) & tr).any()
        # same prompt rendered by other generators is excluded from train
        tr_p = {d.samples[i].prompt_id for i in np.flatnonzero(tr) if d.samples[i].prompt_id}
        te_p = {d.samples[i].prompt_id for i in np.flatnonzero(te) if d.samples[i].prompt_id}
        assert not tr_p & te_p
        assert (d.y[tr] == 1).any() and (d.y[tr] == 0).any()
        covered.setdefault(g, np.zeros(len(d.y), bool))
        covered[g] |= te
    for g, cov in covered.items():
        # pooled folds cover every fake of g and every real exactly
        assert cov[(d.generators == g) & (d.y == 1)].all() and cov[d.y == 0].all()
        assert not cov[(d.y == 1) & (d.generators != g)].any()
    seen = sorted(seen)
    assert seen == ["bing", "dalle", "nano-banana"]


def test_group_by_generator_does_not_lump_reals(tmp_path):
    synth_probs(tmp_path / "m.csv", n_real=10)
    d = FR.load_fit_data([tmp_path / "m.csv"], group_by="generator")
    assert len(set(d.groups[d.y == 0])) == 10
    assert len(set(d.groups[d.y == 1])) == 3


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    z_true = rng.normal(0, 2, 20000)
    y = (rng.uniform(size=z_true.size) < 1 / (1 + np.exp(-z_true))).astype(int)
    assert FR.fit_temperature(3.0 * z_true, y) == pytest.approx(3.0, rel=0.08)


def test_refit_end_to_end_and_backend_compatible_pickle(tmp_path):
    synth_probs(tmp_path / "train.csv", seed=1)
    synth_probs(tmp_path / "test.csv", seed=2, n_prompts=15, n_real=45)
    out = tmp_path / "export"
    rep = FR.refit([tmp_path / "train.csv"], [tmp_path / "test.csv"], out, C_grid=(0.1, 1.0),
                   n_boot=50, seed=0)
    for f in ("fusion_logreg.pkl", "config.json", "label_map.json", "refit_report.json", "test.preds.csv"):
        assert (out / f).exists()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["submodel_order"] == SUBS and cfg["type"] == "probability_stacking_fusion"
    assert cfg["feature_transform"] == "logit" and cfg["temperature"] > 0
    assert rep["settings"]["select_by"] == "logo"
    assert set(rep["grid"][0]["logo"]["per_generator"]) == {"dalle", "nano-banana", "bing"}
    assert max(rep["coef"], key=rep["coef"].get) == "cnn-transfer"
    assert rep["test"]["overall"]["auroc"] > 0.9
    assert any("guessed" in w for w in rep.get("warnings", []))

    X = np.array([[0.9, 0.5, 0.6, 0.7], [0.05, 0.5, 0.4, 0.3], [0.0, 1.0, 0.5, 0.5]])
    import joblib
    expected = joblib.load(out / "fusion_logreg.pkl").predict_proba(X)[:, 1]

    # Loads in a fresh interpreter where this package is not importable (as in the backend).
    code = ("import sys, importlib.util; assert importlib.util.find_spec('eval') is None; "
            "import joblib, numpy as np, json; m = joblib.load(sys.argv[1]); "
            "print(json.dumps(m.predict_proba(np.array(json.loads(sys.argv[2])))[:, 1].tolist()))")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", code, str(out / "fusion_logreg.pkl"), json.dumps(X.tolist())],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    np.testing.assert_allclose(json.loads(r.stdout), expected)

    # Through the real backend wrapper, if its light dependencies are installed.
    pytest.importorskip("pydantic_settings")
    from eval.candidates import import_backend
    import_backend()
    from app.models.wrappers.logreg_fusion_wrapper import LogRegFusionWrapper
    w = LogRegFusionWrapper(repo_id="local", config=cfg, local_path=str(out))
    w.load()
    res = w.predict({k: {"prob_fake": float(v)} for k, v in zip(SUBS, X[0])})
    assert res["prob_fake"] == pytest.approx(expected[0])


def test_refit_grouped_holdout_without_test(tmp_path):
    synth_probs(tmp_path / "train.csv", seed=3)
    rep = FR.refit([tmp_path / "train.csv"], [], tmp_path / "e", C_grid=(1.0,), holdout=0.2, n_boot=20)
    n_test = rep["test"]["overall"]["n"]
    assert 0 < n_test < 240 and rep["n_train"] + n_test == 240


def _manifest_and_probs(tmp_path):
    samples, y, X = synth_probs(tmp_path / "deployed.preds.csv", seed=5)
    write_manifest(samples, tmp_path / "core.csv")
    return samples


def test_run_eval_precomputed_and_offline_fusion_then_compare(tmp_path, capsys):
    _manifest_and_probs(tmp_path)
    base_out = tmp_path / "results" / "base" / "core.json"
    assert run_eval.main(["--candidate", str(tmp_path / "deployed.preds.csv"), "--suite", "core",
                          "--manifest", str(tmp_path / "core.csv"), "--out", str(base_out),
                          "--n-boot", "50", "--name", "base"]) == 0
    res = json.loads(base_out.read_text())
    assert res["schema"] == "deepfake-eval/v1" and res["n_scored"] == 240 and res["n_failed"] == 0
    assert set(res["breakdowns"]["generator"]) == {"dalle", "nano-banana", "bing"}
    assert res["metrics"]["ci95"]["auroc"][0] <= res["metrics"]["auroc"] <= res["metrics"]["ci95"]["auroc"][1]
    assert set(res["submodels"]) == set(SUBS)
    assert (base_out.parent / "core.preds.csv").exists()

    # single submodel column of the same file
    cand = load_candidate(str(tmp_path / "deployed.preds.csv") + "#cnn-transfer")
    assert isinstance(cand, PrecomputedCandidate) and cand.column == "prob.cnn-transfer"

    # refit a fusion, then score it offline through run_eval
    FR.refit([tmp_path / "deployed.preds.csv"], [], tmp_path / "refit", C_grid=(1.0,), holdout=0.2, n_boot=10)
    cand_out = tmp_path / "results" / "refit" / "core.json"
    assert run_eval.main(["--candidate", str(tmp_path / "refit"), "--probs", str(tmp_path / "deployed.preds.csv"),
                          "--suite", "core", "--manifest", str(tmp_path / "core.csv"),
                          "--out", str(cand_out), "--n-boot", "50"]) == 0
    assert isinstance(load_candidate(str(tmp_path / "refit"), probs=str(tmp_path / "deployed.preds.csv")),
                      FusionFileCandidate)

    md = compare.render([str(base_out.parent), str(cand_out.parent)])
    assert "### Suite `core`" in md and "(baseline)" in md
    assert "`base`" in md and "`refit`" in md
    row = [l for l in md.splitlines() if l.startswith("| `refit`")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3].startswith("+")  # refit beats the mean-of-probs baseline
    assert float(cells[4]) < 0.05  # DeLong p from paired preds files
    assert "Per-generator AUROC" in md and "| dalle |" in md
    assert "Metadata-only AUROC" in md

    assert compare.main([str(base_out), str(cand_out), "--out", str(tmp_path / "c.md")]) == 0
    assert (tmp_path / "c.md").read_text(encoding="utf-8").startswith("## Evaluation comparison")


def test_run_eval_missing_images_are_reported(tmp_path):
    samples = _manifest_and_probs(tmp_path)
    t = read_probs(tmp_path / "deployed.preds.csv")
    # drop 5 rows from the probability file
    keep = slice(5, None)
    write_probs(tmp_path / "partial.csv", t.samples()[keep], t.floats("prob_fake")[keep])
    out = tmp_path / "r" / "core.json"
    run_eval.main(["--candidate", str(tmp_path / "partial.csv"), "--suite", "core", "--manifest",
                   str(tmp_path / "core.csv"), "--out", str(out), "--n-boot", "0"])
    res = json.loads(out.read_text())
    assert res["n_failed"] == 5 and res["n_scored"] == len(samples) - 5


def test_bias_probe_suite_through_run_eval(tmp_path):
    rows = []
    for v in ("original", "real_png"):
        for i in range(20):
            lbl = i % 2
            fmt = "png" if (lbl == 1 or v == "real_png") else "jpeg"
            path = f"x/{i}.{'png' if fmt == 'png' else 'jpg'}" if v == "original" or lbl == 1 else f"_bias_probe/{v}/{i}.png"
            rows.append(Sample(path, lbl, "g" if lbl else "real", "s", fmt, 64, 64, "",
                               {"variant": v, "orig_path": f"x/{i}.{'png' if lbl else 'jpg'}"}))
    write_manifest(rows, tmp_path / "bias-probe.csv")
    probs = np.array([0.9 if r.format == "png" else 0.1 for r in rows])
    write_probs(tmp_path / "p.csv", rows, probs)
    out = tmp_path / "r" / "bias-probe.json"
    run_eval.main(["--candidate", str(tmp_path / "p.csv"), "--suite", "bias-probe", "--manifest",
                   str(tmp_path / "bias-probe.csv"), "--out", str(out), "--n-boot", "20"])
    res = json.loads(out.read_text())
    assert res["n_scored"] == 20  # headline metrics on originals only
    assert res["bias_probe"]["variants"]["real_png"]["auroc_drop"] == pytest.approx(0.5)
    md = compare.render([str(out)])
    assert "Bias probe" in md and "drop +0.5000" in md
