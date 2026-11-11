"""In-process backend adapters, exercised offline with stub HF repos (no network, no weights)."""

import json

import numpy as np
import pytest

pytest.importorskip("pydantic_settings")

from eval import run_eval  # noqa: E402
from eval.candidates import import_backend, load_candidate  # noqa: E402
from eval.datasets import scan_split_layout, write_manifest  # noqa: E402

PREDICT_PY = """
def predict(image_bytes=None, **kwargs):
    # deterministic: PNG files look 'fake'
    p = {offset} + (0.5 if image_bytes[:4] == b"\\x89PNG" else 0.0)
    return {{"pred_int": int(p >= 0.5), "prob_fake": p}}
"""


@pytest.fixture
def stub_hub(tmp_path, monkeypatch):
    import_backend()
    import app.services.model_registry as reg_mod
    import app.services.hf_hub_service as hub_mod

    repos = {}
    for name, off in (("sub-a", 0.2), ("sub-b", 0.3)):
        d = tmp_path / "hub" / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"name": name}))  # -> DummyRandomWrapper
        (d / "predict.py").write_text(PREDICT_PY.format(offset=off))
        repos[f"local/{name}"] = d
    fusion = tmp_path / "hub" / "fusion"
    fusion.mkdir()
    (fusion / "config.json").write_text(json.dumps({
        "type": "probability_stacking_fusion", "submodels": ["local/sub-a", "local/sub-b"],
        "submodel_order": ["sub-a", "sub-b"], "threshold": 0.5}))
    import joblib
    from sklearn.linear_model import LogisticRegression

    X = np.array([[0.2, 0.3], [0.7, 0.8], [0.25, 0.3], [0.7, 0.75]])
    joblib.dump(LogisticRegression().fit(X, [0, 1, 0, 1]), fusion / "fusion_logreg.pkl")
    repos["local/fusion"] = fusion

    class StubHub:
        def download_repo(self, repo_id, **kw):
            return str(repos[repo_id])

    monkeypatch.setattr(reg_mod, "get_hf_hub_service", lambda: StubHub())
    monkeypatch.setattr(hub_mod, "get_hf_hub_service", lambda: StubHub())
    return repos


def test_deployed_stack_in_process(stub_hub, split_tree, tmp_path):
    root, split = split_tree
    write_manifest(scan_split_layout(split, root=root), tmp_path / "core.csv")
    out = tmp_path / "res" / "core.json"
    assert run_eval.main(["--candidate", "deployed@local/fusion", "--suite", "core", "--manifest",
                          str(tmp_path / "core.csv"), "--data-root", str(root), "--out", str(out),
                          "--n-boot", "20", "--name", "stub"]) == 0
    res = json.loads(out.read_text())
    assert res["n_scored"] == 20 and res["n_failed"] == 0
    assert res["metrics"]["auroc"] == 1.0  # PNG fakes vs JPEG reals: the shortcut the probe targets
    assert res["metadata_shortcut_auroc"]["format"] == 1.0
    assert set(res["submodels"]) == {"sub-a", "sub-b"}
    assert res["candidate"]["fusion_repo"] == "local/fusion"
    preds = (out.parent / "core.preds.csv").read_text().splitlines()
    assert "prob.sub-a" in preds[0] and "prob.sub-b" in preds[0]


def test_single_submodel_in_process(stub_hub, split_tree):
    root, split = split_tree
    samples = scan_split_layout(split, root=root)
    cand = load_candidate("submodel:sub-b@local/fusion")
    preds = cand.predict(samples, lambda s: root / s.path)
    expected = np.array([0.8 if s.format == "png" else 0.3 for s in samples])
    np.testing.assert_allclose(preds.prob_fake, expected)
    assert cand.info()["repo_id"] == "local/sub-b"


def test_unreadable_image_is_a_reported_failure(stub_hub, split_tree):
    root, split = split_tree
    samples = scan_split_layout(split, root=root)
    samples[0].path = "does/not/exist.png"
    cand = load_candidate("deployed@local/fusion")
    preds = cand.predict(samples, lambda s: root / s.path)
    assert 0 in preds.errors and np.isnan(preds.prob_fake[0]) and np.isfinite(preds.prob_fake[1:]).all()
