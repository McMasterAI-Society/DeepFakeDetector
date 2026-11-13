"""Fusion contribution tests (no downloads; a LogisticRegression fit in-test)."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.models.wrappers.logreg_fusion_wrapper import LogRegFusionWrapper
from app.schemas.predict import FusionMeta

ORDER = ["cnn-transfer", "vit-base", "deit-distilled", "gradfield-cnn"]


def _fit(transform="prob", pipeline=False, negative=False):
    rng = np.random.default_rng(0)
    X = rng.random((200, 4))
    y = (X @ np.array([3.0, 1.0, -2.0 if negative else 0.5, 0.2]) > 1.2).astype(int)
    if transform == "logit":
        p = np.clip(X, 1e-6, 1 - 1e-6)
        X = np.log(p / (1 - p))
    est = LogisticRegression(max_iter=1000)
    model = make_pipeline(StandardScaler(), est) if pipeline else est
    return model.fit(X, y)


def _wrapper(model, **config):
    w = LogRegFusionWrapper("x/fusion", {"submodel_order": ORDER, **config}, ".")
    w._model = model
    return w


def _outputs(probs):
    return {n: {"prob_fake": p} for n, p in zip(ORDER, probs)}


def _logit(p):
    return np.log(p / (1 - p))


@pytest.mark.parametrize("transform,pipeline,negative", [
    ("prob", False, False),
    ("prob", False, True),
    ("logit", False, False),
    ("prob", True, True),
])
def test_contributions_are_exact_and_sum_to_100(transform, pipeline, negative):
    w = _wrapper(_fit(transform, pipeline, negative), feature_transform=transform)
    probs = [0.9, 0.3, 0.7, 0.1]
    out = w.predict(_outputs(probs))
    meta = out["meta"]

    pct = meta["contribution_percentages"]
    assert set(pct) == set(ORDER)
    assert sum(pct.values()) == pytest.approx(100.0)
    assert all(v >= 0 for v in pct.values())

    wc = meta["weighted_contributions"]
    # intercept + sum(contributions) reproduces the model's log-odds
    assert meta["intercept"] + sum(wc.values()) == pytest.approx(_logit(out["prob_fake"]), abs=1e-6)
    assert meta["fusion_logit"] == pytest.approx(_logit(out["prob_fake"]), abs=1e-6)

    for n in ORDER:
        expected = "fake" if wc[n] > 0 else "real" if wc[n] < 0 else "neutral"
        assert meta["contribution_directions"][n] == expected
        assert pct[n] == pytest.approx(abs(wc[n]) / sum(abs(v) for v in wc.values()) * 100)

    if not pipeline:
        x = np.asarray(probs)
        if transform == "logit":
            x = _logit(x)
        coef = w._model.coef_[0]
        for i, n in enumerate(ORDER):
            assert meta["submodel_weights"][n] == pytest.approx(coef[i])
            assert wc[n] == pytest.approx(coef[i] * x[i])

    # Backward compatible with the response schema
    fm = FusionMeta(**{k: meta[k] for k in (
        "submodel_weights", "weighted_contributions", "contribution_percentages",
        "contribution_directions", "intercept")})
    assert fm.contribution_percentages == pct


def test_negative_coefficient_reports_real_direction():
    w = _wrapper(_fit(negative=True))
    meta = w.predict(_outputs([0.5, 0.5, 0.9, 0.5]))["meta"]
    assert w._model.coef_[0][2] < 0
    assert meta["contribution_directions"]["deit-distilled"] == "real"


def test_contribution_baseline_shifts_into_intercept():
    model = _fit()
    plain = _wrapper(model).predict(_outputs([0.2, 0.4, 0.6, 0.8]))
    based = _wrapper(model, contribution_baseline={n: 0.5 for n in ORDER}).predict(
        _outputs([0.2, 0.4, 0.6, 0.8]))
    assert based["prob_fake"] == plain["prob_fake"]
    m = based["meta"]
    assert m["intercept"] + sum(m["weighted_contributions"].values()) == pytest.approx(m["fusion_logit"])
    # cnn-transfer is below the 0.5 reference with a positive weight -> toward real
    assert m["contribution_directions"]["cnn-transfer"] == "real"


def test_all_zero_contributions_split_evenly():
    model = _fit()
    meta = _wrapper(model).predict(_outputs([0.0, 0.0, 0.0, 0.0]))["meta"]
    assert all(v == pytest.approx(25.0) for v in meta["contribution_percentages"].values())


def test_prediction_survives_model_without_coef():
    class Opaque:
        def predict_proba(self, X):
            return np.array([[0.4, 0.6]])

    out = _wrapper(Opaque()).predict(_outputs([0.1, 0.2, 0.3, 0.4]))
    assert out["prob_fake"] == pytest.approx(0.6)
    assert "contribution_percentages" not in out["meta"]


def test_clip_logit_pipeline_from_eval_harness():
    """Refit fusions are exported as Pipeline(clip -> logit -> LogisticRegression)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    eps = 1e-4
    rng = np.random.default_rng(0)
    X = rng.random((300, 4))
    y = (X @ np.array([3.0, 1.0, -1.0, 0.2]) > 1.3).astype(int)
    pipe = Pipeline([
        ("clip", FunctionTransformer(lambda a: np.clip(a, eps, 1 - eps))),
        ("logit", FunctionTransformer(lambda a: np.log(a / (1 - a)))),
        ("lr", LogisticRegression(max_iter=1000)),
    ]).fit(X, y)

    # Features stay raw probabilities at the wrapper level: the pipeline does the logit
    w = _wrapper(pipe, temperature=1.3)
    probs = [0.95, 0.4, 0.8, 0.00001]
    out = w.predict(_outputs(probs))
    meta = out["meta"]

    lr = pipe[-1]
    z = np.log(np.clip(probs, eps, 1 - eps) / (1 - np.clip(probs, eps, 1 - eps)))
    for i, n in enumerate(ORDER):
        assert meta["submodel_weights"][n] == pytest.approx(lr.coef_[0][i])
        assert meta["weighted_contributions"][n] == pytest.approx(lr.coef_[0][i] * z[i])
    assert meta["intercept"] == pytest.approx(lr.intercept_[0])
    assert meta["fusion_logit"] == pytest.approx(_logit(out["prob_fake"]), abs=1e-6)
    assert sum(meta["contribution_percentages"].values()) == pytest.approx(100.0)
    assert meta["temperature"] == pytest.approx(1.3)
    # logit features can be negative, so a branch can push toward real
    assert meta["contribution_directions"]["gradfield-cnn"] == ("real" if lr.coef_[0][3] > 0 else "fake")


def test_falls_back_to_config_coefficients():
    class Opaque:
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])

    coef = {"cnn-transfer": 2.0, "vit-base": -1.0, "deit-distilled": 0.5, "gradfield-cnn": 0.0}
    meta = _wrapper(Opaque(), coef=coef, intercept=-0.25).predict(_outputs([0.5, 0.5, 0.2, 0.9]))["meta"]
    assert meta["weighted_contributions"] == pytest.approx(
        {"cnn-transfer": 1.0, "vit-base": -0.5, "deit-distilled": 0.1, "gradfield-cnn": 0.0})
    assert meta["intercept"] == pytest.approx(-0.25)
    assert meta["contribution_directions"]["vit-base"] == "real"
    assert meta["contribution_directions"]["gradfield-cnn"] == "neutral"
    assert sum(meta["contribution_percentages"].values()) == pytest.approx(100.0)
