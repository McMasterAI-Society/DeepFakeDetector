"""Colormap LUT and faithfulness-metric tests."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.services import explain_metrics as em
from app.services.colormaps import DIVERGING_LUT, TURBO_LUT, apply_colormap, get_lut
from app.services.explainability import compute_cam


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

def test_lut_shapes_and_known_entries():
    for lut in (TURBO_LUT, DIVERGING_LUT):
        assert lut.shape == (256, 3) and lut.dtype == np.uint8
        assert not lut.flags.writeable
    assert tuple(TURBO_LUT[0]) == (48, 18, 59)      # turbo starts dark purple
    assert tuple(TURBO_LUT[255]) == (122, 4, 2)     # and ends dark red
    assert tuple(DIVERGING_LUT[0]) == (5, 48, 97)   # RdBu_r: blue ...
    assert tuple(DIVERGING_LUT[255]) == (103, 0, 31)  # ... to red
    assert DIVERGING_LUT[127].min() >= 240           # white centre


def test_apply_colormap_indexing_and_edge_values():
    v = np.array([[0.0, 1.0, -0.5, 2.0], [np.nan, 0.5, 1 / 256 - 1e-9, 1 / 256]])
    out = apply_colormap(v, "turbo")
    assert out.shape == (2, 4, 3) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == tuple(TURBO_LUT[0])
    assert tuple(out[0, 1]) == tuple(TURBO_LUT[255])
    assert tuple(out[0, 2]) == tuple(TURBO_LUT[0])     # clipped
    assert tuple(out[0, 3]) == tuple(TURBO_LUT[255])   # clipped
    assert tuple(out[1, 0]) == tuple(TURBO_LUT[0])     # NaN -> 0
    assert tuple(out[1, 1]) == tuple(TURBO_LUT[128])
    assert tuple(out[1, 2]) == tuple(TURBO_LUT[0])
    assert tuple(out[1, 3]) == tuple(TURBO_LUT[1])
    assert get_lut("unknown") is TURBO_LUT


@pytest.mark.parametrize("ours,theirs", [("turbo", "turbo"), ("diverging", "RdBu_r")])
def test_lut_matches_matplotlib_exactly(ours, theirs):
    matplotlib = pytest.importorskip("matplotlib")
    x = np.random.default_rng(0).random((64, 64))
    x[0, :3] = [0.0, 1.0, 0.5]
    ref = (matplotlib.colormaps[theirs](x)[..., :3] * 255).astype(np.uint8)
    np.testing.assert_array_equal(apply_colormap(x, ours), ref)


def test_explainability_does_not_import_matplotlib():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "app"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] == "matplotlib" for n in names), path


# ---------------------------------------------------------------------------
# Deletion / insertion
# ---------------------------------------------------------------------------

def _region_predict(batch):
    """Score = mean of channel 0 inside the top-left 4x4 block."""
    return batch[:, 0, :4, :4].mean(dim=(1, 2))


def test_deletion_insertion_rewards_the_right_map():
    x = torch.zeros(1, 1, 8, 8)
    x[0, 0, :4, :4] = 1.0
    good = np.zeros((8, 8), dtype=np.float32)
    good[:4, :4] = 1.0
    bad = 1.0 - good

    g = em.deletion_insertion(_region_predict, x, good, steps=16, baseline="zero")
    b = em.deletion_insertion(_region_predict, x, bad, steps=16, baseline="zero")
    assert len(g["deletion"]) == 17 and g["fractions"][0] == 0 and g["fractions"][-1] == 1
    assert g["deletion"][0] == pytest.approx(1.0) and g["deletion"][-1] == pytest.approx(0.0)
    assert g["insertion"][0] == pytest.approx(0.0) and g["insertion"][-1] == pytest.approx(1.0)
    assert g["deletion_auc"] < b["deletion_auc"]
    assert g["insertion_auc"] > b["insertion_auc"]
    r = em.deletion_insertion(_region_predict, x, em.random_heatmap((8, 8)), steps=16, baseline="zero")
    assert g["deletion_auc"] <= r["deletion_auc"] <= b["deletion_auc"]


def test_heatmap_is_resized_and_baselines_work():
    x = torch.rand(1, 3, 16, 16)
    hm = np.random.default_rng(1).random((4, 4))
    for base in ("blur", "zero", "mean"):
        out = em.deletion_insertion(lambda b: b.mean(dim=(1, 2, 3)), x, hm, steps=5, baseline=base, batch_size=4)
        assert len(out["insertion"]) == 6
    assert em.make_baseline(x, "blur").shape == x.shape


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(self.features(x).mean(dim=(2, 3)))


def test_spearman_basics():
    a = np.arange(10.0)
    assert em.spearman(a, a) == pytest.approx(1.0)
    assert em.spearman(a, -a) == pytest.approx(-1.0)
    assert em.spearman(a, np.zeros(10)) == 0.0


def test_randomization_check_separates_model_dependent_maps():
    torch.manual_seed(0)
    model = TinyCNN().eval()
    x = torch.randn(1, 3, 16, 16)

    def cam_fn(m, xx):
        return compute_cam(m, xx, m.features[3], method="hirescam", target_class=1).signed_map

    def input_only_fn(_m, xx):
        # Ignores the model entirely: an edge detector "explanation"
        return xx[0].abs().sum(0).numpy()

    weights_before = [p.detach().clone() for p in model.parameters()]
    cam = em.parameter_randomization_check(cam_fn, model, x, seed=3)
    edge = em.parameter_randomization_check(input_only_fn, model, x, seed=3)
    assert edge["final_spearman"] == pytest.approx(1.0)
    assert cam["final_spearman"] < 0.5
    # The original model is untouched
    for before, p in zip(weights_before, model.parameters()):
        assert torch.equal(before, p)

    cascade = em.parameter_randomization_check(cam_fn, model, x, cascade=["fc", "features.2", "features.0"])
    assert [s["randomized"] for s in cascade["steps"]] == [["fc"], ["fc", "features.2"], ["fc", "features.2", "features.0"]]
    with pytest.raises(KeyError):
        em.parameter_randomization_check(cam_fn, model, x, cascade=["nope"])
