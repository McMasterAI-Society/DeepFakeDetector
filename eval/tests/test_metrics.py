import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score, roc_curve

from eval import metrics as M


def notebook_ece(probs, labels, n_bins=15):
    """Copy of compute_ece from notebooks/fusion-logreg.ipynb."""
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        ece += (n_in_bin / len(probs)) * abs(labels[mask].mean() - probs[mask].mean())
    return ece


def test_auroc_matches_sklearn_with_ties(synthetic_scores):
    y, p = synthetic_scores
    assert M.auroc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_roc_points_match_sklearn(synthetic_scores):
    y, p = synthetic_scores
    fpr, tpr, thr = M.roc_points(y, p)
    sf, st, sthr = roc_curve(y, p, drop_intermediate=False)
    np.testing.assert_allclose(fpr, sf)
    np.testing.assert_allclose(tpr, st)
    np.testing.assert_allclose(thr[1:], sthr[1:])


@pytest.mark.parametrize("target", [0.01, 0.05, 0.2])
def test_tpr_at_fpr_vs_sklearn(synthetic_scores, target):
    y, p = synthetic_scores
    sf, st, _ = roc_curve(y, p, drop_intermediate=False)
    assert M.tpr_at_fpr(y, p, target) == pytest.approx(st[sf <= target].max())


@pytest.mark.parametrize("target", [0.5, 0.9, 0.95])
def test_fpr_at_tpr_vs_sklearn(synthetic_scores, target):
    y, p = synthetic_scores
    sf, st, _ = roc_curve(y, p, drop_intermediate=False)
    assert M.fpr_at_tpr(y, p, target) == pytest.approx(sf[st >= target].min())


@pytest.mark.parametrize("thr", [0.3, 0.5, 0.7])
def test_balanced_accuracy_and_brier(synthetic_scores, thr):
    y, p = synthetic_scores
    assert M.balanced_accuracy(y, p, thr) == pytest.approx(balanced_accuracy_score(y, (p >= thr).astype(int)))
    assert M.brier(y, p) == pytest.approx(brier_score_loss(y, p))


def test_ece_matches_notebook(synthetic_scores):
    y, p = synthetic_scores
    assert M.ece(y, p, 15) == pytest.approx(notebook_ece(p, y, 15), abs=1e-12)
    # edge values land in the first / last bin
    y2 = np.array([0, 1, 1, 0])
    p2 = np.array([0.0, 1.0, 1 / 15, 0.5])
    assert M.ece(y2, p2) == pytest.approx(notebook_ece(p2, y2))


def test_perfectly_calibrated_has_low_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert M.ece(y, p) < 0.01


def test_single_class_is_none():
    y = np.ones(10, int)
    s = np.linspace(0, 1, 10)
    assert M.auroc(y, s) is None
    assert M.tpr_at_fpr(y, s) is None
    assert M.balanced_accuracy(y, s) is None
    assert M.delong_test(y, s, s) is None
    assert M.bootstrap_ci(y, s, n_boot=10)["auroc"] is None


def test_bootstrap_ci_contains_point_estimate(synthetic_scores):
    y, p = synthetic_scores
    ci = M.bootstrap_ci(y, p, n_boot=300, seed=1)
    for k in M.CORE_METRICS:
        lo, hi = ci[k]
        point = M.compute_metrics(y, p)[k]
        assert lo <= point + 1e-9 and point - 1e-9 <= hi, k
    # deterministic for a fixed seed
    assert M.bootstrap_ci(y, p, n_boot=50, seed=7) == M.bootstrap_ci(y, p, n_boot=50, seed=7)


def test_delong_identical_predictions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.uniform(size=200) + 0.3 * y
    t = M.delong_test(y, s, s)
    assert t["diff"] == 0 and t["p_value"] == 1.0
    assert t["auroc_1"] == pytest.approx(roc_auc_score(y, s))


def test_delong_detects_large_difference_and_not_noise():
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, n)
    strong = y * 2.0 + rng.normal(size=n)
    weak = y * 0.2 + rng.normal(size=n)
    t = M.delong_test(y, strong, weak)
    assert t["diff"] > 0.2 and t["p_value"] < 1e-6
    # two equally good but independent noisy scorers: usually not significant
    a = y * 1.0 + rng.normal(size=n)
    b = y * 1.0 + rng.normal(size=n)
    assert M.delong_test(y, a, b)["p_value"] > 0.01
    # antisymmetric
    t2 = M.delong_test(y, weak, strong)
    assert t2["diff"] == pytest.approx(-t["diff"]) and t2["p_value"] == pytest.approx(t["p_value"])


def test_delong_se_agrees_with_bootstrap():
    rng = np.random.default_rng(2)
    n = 400
    y = rng.integers(0, 2, n)
    s = y * 1.0 + rng.normal(size=n)
    se = M.delong_ci(y, s)["se"]
    boots = []
    for _ in range(1500):
        i = rng.integers(0, n, n)
        if len(set(y[i])) == 2:
            boots.append(roc_auc_score(y[i], s[i]))
    assert se == pytest.approx(np.std(boots), rel=0.15)


def test_breakdowns_and_shortcut():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.9, 0.8, 0.4, 0.7])
    gen = ["real", "real", "real", "a", "a", "b", "b"]
    b = M.breakdown(y, s, gen, 0.5, fakes_vs_all_reals=True, real_key="real")
    assert set(b) == {"a", "b"}
    assert b["a"]["n_real"] == 3 and b["a"]["n_fake"] == 2
    assert b["a"]["auroc"] == pytest.approx(roc_auc_score([0, 0, 0, 1, 1], [0.1, 0.2, 0.6, 0.9, 0.8]))
    assert b["b"]["tpr"] == pytest.approx(0.5)
    fmt = ["jpeg"] * 3 + ["png"] * 4
    assert M.metadata_shortcut_auroc(y, fmt) == 1.0
    sub = M.breakdown(y, s, fmt, 0.5)
    assert sub["jpeg"]["auroc"] is None and sub["jpeg"]["tnr"] == pytest.approx(2 / 3)
