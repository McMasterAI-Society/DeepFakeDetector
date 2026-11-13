"""
Faithfulness and sanity metrics for saliency maps.

Model-agnostic: everything works on a batched `predict_fn` (input batch ->
score per item, typically the probability of the explained class) and a 2D
heatmap aligned with the input's spatial dimensions.

- Deletion / insertion curves and AUC (Petsiuk et al., 2018, "RISE"):
  remove (or reveal) the most salient pixels first and track the score.
  A faithful map gives a LOW deletion AUC and a HIGH insertion AUC.
- Model-parameter randomisation sanity check (Adebayo et al., 2018,
  "Sanity Checks for Saliency Maps"): a map that barely changes when the
  weights are randomised does not depend on what the model learned.
"""

import copy
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# numpy 2 deprecates trapz in favour of trapezoid (absent in numpy < 2)
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

PredictFn = Callable[[torch.Tensor], Union[torch.Tensor, np.ndarray, Sequence[float]]]
ExplainFn = Callable[[nn.Module, torch.Tensor], np.ndarray]


def _to_numpy_scores(scores) -> np.ndarray:
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().float().cpu().numpy()
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def make_baseline(x: torch.Tensor, kind: Union[str, torch.Tensor] = "blur") -> torch.Tensor:
    """
    Reference input that pixels are replaced with (deletion) or revealed
    from (insertion).

    kind: "blur" (Gaussian blur, removes high-frequency artifacts while
    keeping colour and layout), "zero" (zeros in the model's input space,
    i.e. the dataset mean after normalisation), "mean" (per-channel image
    mean), or an explicit tensor shaped like x.
    """
    if isinstance(kind, torch.Tensor):
        return kind.to(x.device, x.dtype).expand_as(x)
    if kind == "zero":
        return torch.zeros_like(x)
    if kind == "mean":
        return x.mean(dim=(-2, -1), keepdim=True).expand_as(x).clone()
    if kind == "blur":
        sigma = max(x.shape[-1], x.shape[-2]) / 32.0
        radius = int(3 * sigma)
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        k1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1 = k1 / k1.sum()
        c = x.shape[1]
        kx = k1.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        ky = k1.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
        y = F.pad(x, (radius, radius, 0, 0), mode="reflect")
        y = F.conv2d(y, kx, groups=c)
        y = F.pad(y, (0, 0, radius, radius), mode="reflect")
        return F.conv2d(y, ky, groups=c)
    raise ValueError(f"Unknown baseline: {kind!r}")


def _pixel_order(heatmap: np.ndarray, size) -> torch.Tensor:
    """Pixel indices sorted from most to least salient (ties broken stably)."""
    h = torch.as_tensor(np.asarray(heatmap, dtype=np.float32))
    if tuple(h.shape) != tuple(size):
        h = F.interpolate(h[None, None], size=tuple(size), mode="bilinear", align_corners=False)[0, 0]
    return torch.argsort(h.flatten(), descending=True, stable=True)


def deletion_insertion(
    predict_fn: PredictFn,
    x: torch.Tensor,
    heatmap: np.ndarray,
    steps: int = 20,
    baseline: Union[str, torch.Tensor] = "blur",
    batch_size: int = 16,
) -> Dict[str, object]:
    """
    Deletion and insertion curves for one input and one heatmap.

    Args:
        predict_fn: maps a batch [B, C, H, W] to B scores (e.g. probability
            of the class the heatmap explains)
        x: input [1, C, H, W] in the model's input space
        heatmap: 2D saliency (any scale; only the ranking matters); resized
            to x's spatial size if needed
        steps: number of perturbation steps (curve has steps + 1 points)
        baseline: see `make_baseline`
        batch_size: forward batch size

    Returns:
        dict with fractions, deletion/insertion scores, and their AUCs
        (trapezoid rule over fraction in [0, 1]).
    """
    if x.dim() != 4 or x.shape[0] != 1:
        raise ValueError("x must have shape [1, C, H, W]")
    H, W = x.shape[-2:]
    n_pix = H * W
    order = _pixel_order(heatmap, (H, W)).to(x.device)
    base = make_baseline(x, baseline)

    fractions = np.linspace(0.0, 1.0, steps + 1)
    # rank[p] = position of pixel p in the saliency order
    rank = torch.empty(n_pix, dtype=torch.long, device=x.device)
    rank[order] = torch.arange(n_pix, device=x.device)

    def curve(start, end) -> np.ndarray:
        scores: List[float] = []
        batch = []
        for f in fractions:
            k = int(round(f * n_pix))
            top = (rank < k).view(1, 1, H, W).to(x.dtype)
            batch.append(start * (1 - top) + end * top)
            if len(batch) == batch_size:
                with torch.no_grad():
                    scores.extend(_to_numpy_scores(predict_fn(torch.cat(batch))))
                batch = []
        if batch:
            with torch.no_grad():
                scores.extend(_to_numpy_scores(predict_fn(torch.cat(batch))))
        return np.asarray(scores, dtype=np.float64)

    deletion = curve(x, base)    # salient pixels go to the baseline first
    insertion = curve(base, x)   # salient pixels are revealed first
    return {
        "fractions": fractions.tolist(),
        "deletion": deletion.tolist(),
        "insertion": insertion.tolist(),
        "deletion_auc": float(_trapezoid(deletion, fractions)),
        "insertion_auc": float(_trapezoid(insertion, fractions)),
    }


def random_heatmap(shape, seed: int = 0) -> np.ndarray:
    """Uniform random map; a control that any useful method should beat."""
    return np.random.default_rng(seed).random(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _rank(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (like scipy.stats.rankdata(method="average"))."""
    sorter = np.argsort(a, kind="stable")
    inv = np.empty(len(a), dtype=np.intp)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    first = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = first.cumsum()[inv]
    bounds = np.r_[np.nonzero(first)[0], len(a)]
    return 0.5 * (bounds[dense] + bounds[dense - 1] + 1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two maps (flattened); 0 if either is constant."""
    ra = _rank(np.asarray(a, dtype=np.float64).ravel())
    rb = _rank(np.asarray(b, dtype=np.float64).ravel())
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def randomize_parameters(module: nn.Module, seed: int = 0) -> nn.Module:
    """Re-initialise every parameter of `module` in place (deterministically)."""
    gen_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        reset = set()
        for m in module.modules():
            if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
                m.reset_parameters()
                reset.update(id(p) for p in m.parameters(recurse=False))
            # BatchNorm running stats are buffers, not parameters
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats()
        # Parameters without a reset_parameters owner (e.g. cls_token,
        # pos_embed, layer-scale gammas) still need to change.
        with torch.no_grad():
            for p in module.parameters(recurse=True):
                if id(p) not in reset:
                    p.normal_(0.0, 0.02)
    finally:
        torch.random.set_rng_state(gen_state)
    return module


def parameter_randomization_check(
    explain_fn: ExplainFn,
    model: nn.Module,
    x: torch.Tensor,
    cascade: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Compare a model's map with maps from randomised copies of the model.

    Args:
        explain_fn: (model, x) -> 2D heatmap
        model: the trained model (left untouched; copies are randomised)
        x: input [1, C, H, W]
        cascade: optional submodule names ordered from output to input. Each
            step randomises one more of them (cascading randomisation). If
            omitted, all parameters are randomised at once.

    Returns:
        dict with Spearman correlations (signed and of absolute values) for
        each randomisation step. Lower is better: a map that still matches
        after randomisation is not explaining the learned weights.
    """
    reference = np.asarray(explain_fn(model, x), dtype=np.float32)
    steps = []
    work = copy.deepcopy(model).eval()

    if cascade:
        named = dict(work.named_modules())
        for i, name in enumerate(cascade):
            if name not in named:
                raise KeyError(f"No submodule named {name!r}")
            randomize_parameters(named[name], seed=seed + i)
            hm = np.asarray(explain_fn(work, x), dtype=np.float32)
            steps.append({
                "randomized": list(cascade[: i + 1]),
                "spearman": spearman(reference, hm),
                "spearman_abs": spearman(np.abs(reference), np.abs(hm)),
            })
    else:
        randomize_parameters(work, seed=seed)
        hm = np.asarray(explain_fn(work, x), dtype=np.float32)
        steps.append({
            "randomized": ["<all>"],
            "spearman": spearman(reference, hm),
            "spearman_abs": spearman(np.abs(reference), np.abs(hm)),
        })

    return {"steps": steps, "final_spearman": steps[-1]["spearman"]}
