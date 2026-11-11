"""
Explainability utilities for DeepFake detection models.

Provides:
- CNN class activation maps (HiResCAM, Grad-CAM, Grad-CAM++). The logits and
  the map come from one forward pass plus one partial backward pass.
- Class-specific gradient-weighted attention rollout for ViT/DeiT, following
  Chefer et al. 2021 ("Generic Attention-model Explainability for Interpreting
  Bi-Modal and Encoder-Decoder Transformers"). It uses the attention
  probabilities timm actually computes (q_norm/k_norm included). Plain
  class-agnostic attention rollout is kept as a fallback method.
- Signed maps ("evidence for fake" vs. "evidence for real").
- Heatmap encoding with numpy colormap lookup tables (no matplotlib).
- A short text summary of where a heatmap concentrates.
"""

import base64
import io
import math
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.core.logging import get_logger
from app.services.colormaps import apply_colormap

logger = get_logger(__name__)

TargetSpec = Union[int, Callable[[torch.Tensor], int], None]
LogitsFn = Callable[[Any], torch.Tensor]

CAM_METHODS = ("hirescam", "gradcam", "gradcam++")
TRANSFORMER_METHODS = ("grad_rollout", "rollout")


# ---------------------------------------------------------------------------
# Scores, normalisation, resizing
# ---------------------------------------------------------------------------

def default_logits_fn(output: Any) -> torch.Tensor:
    """Extract a [B, K] logits tensor from a model output."""
    logits = output[0] if isinstance(output, (tuple, list)) else output
    if logits.dim() == 1:
        # Single-logit binary models return shape [B]
        logits = logits.unsqueeze(-1)
    return logits


def predicted_class(logits: torch.Tensor, threshold: float = 0.5) -> int:
    """Predicted class index for [1, K] logits (K == 1 means sigmoid)."""
    if logits.shape[-1] == 1:
        return int(torch.sigmoid(logits[0, 0]).item() >= threshold)
    if logits.shape[-1] == 2:
        return int(torch.softmax(logits[0], dim=-1)[1].item() >= threshold)
    return int(logits[0].argmax().item())


def class_score(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    """
    Scalar score whose gradient explains `target_class`.

    Uses the log-odds of the target class rather than its raw logit:
    - K == 1 (sigmoid): +logit for class 1, -logit for class 0
    - K >= 2 (softmax): logit[c] - logsumexp(logits of the other classes),
      which for K == 2 is logit[c] - logit[1 - c].

    The log-odds is what the softmax decision depends on; the raw logit also
    carries the component shared by both classes, which cancels out.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.shape[-1] == 1:
        s = logits[0, 0]
        return s if target_class == 1 else -s
    others = torch.cat([logits[0, :target_class], logits[0, target_class + 1:]])
    return logits[0, target_class] - torch.logsumexp(others, dim=0)


def _resolve_target(target: TargetSpec, logits: torch.Tensor) -> int:
    if target is None:
        return predicted_class(logits)
    if callable(target):
        return int(target(logits))
    return int(target)


def normalize_map(m: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Min-max normalise a map to [0, 1] (all-constant maps become zeros)."""
    if isinstance(m, torch.Tensor):
        m = m.detach().float().cpu().numpy()
    m = np.nan_to_num(np.asarray(m, dtype=np.float32))
    m = m - m.min() if m.size else m
    mx = m.max() if m.size else 0.0
    return (m / mx).astype(np.float32) if mx > 0 else np.zeros_like(m, dtype=np.float32)


def normalize_signed(m: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Scale a signed map to [-1, 1] by its largest absolute value (0 stays 0)."""
    if isinstance(m, torch.Tensor):
        m = m.detach().float().cpu().numpy()
    m = np.nan_to_num(np.asarray(m, dtype=np.float32))
    mx = np.abs(m).max() if m.size else 0.0
    return (m / mx).astype(np.float32) if mx > 0 else np.zeros_like(m, dtype=np.float32)


def resize_map(m: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a float (possibly signed) 2D map to (H, W)."""
    m = np.asarray(m, dtype=np.float32)
    if tuple(m.shape) == tuple(size):
        return m
    t = torch.from_numpy(np.ascontiguousarray(m))[None, None]
    t = F.interpolate(t, size=tuple(size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


# ---------------------------------------------------------------------------
# Hook utilities
# ---------------------------------------------------------------------------

_LOCKS_GUARD = threading.Lock()
_MODEL_LOCKS: "weakref.WeakKeyDictionary[nn.Module, threading.RLock]" = weakref.WeakKeyDictionary()


def _model_lock(model: nn.Module) -> threading.RLock:
    """Per-model lock so explain passes never interleave their hooks/flags."""
    with _LOCKS_GUARD:
        lock = _MODEL_LOCKS.get(model)
        if lock is None:
            lock = threading.RLock()
            _MODEL_LOCKS[model] = lock
        return lock


@contextmanager
def capture_outputs(modules: Sequence[nn.Module]) -> Iterator[List[torch.Tensor]]:
    """
    Record the forward outputs (graph attached) of `modules`, in call order.

    Only forward passes on the calling thread are recorded, so a concurrent
    plain prediction on the same model does not pollute the capture.
    """
    owner = threading.get_ident()
    storage: List[torch.Tensor] = []

    def hook(_module, _inputs, output):
        if threading.get_ident() == owner:
            storage.append(output[0] if isinstance(output, (tuple, list)) else output)

    handles = [m.register_forward_hook(hook) for m in modules]
    try:
        yield storage
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# CNN class activation maps
# ---------------------------------------------------------------------------

def cam_from_activations(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    method: str = "hirescam",
) -> torch.Tensor:
    """
    Combine activations and gradients [1, C, h, w] into a signed map [h, w].

    - hirescam:  sum_c(dS/dA * A), element-wise (Draelos & Carin, 2020).
                 For a layer followed only by global average pooling and
                 linear layers this sums exactly to (score - constant).
    - gradcam:   sum_c(mean_hw(dS/dA) * A) (Selvaraju et al., 2017).
    - gradcam++: Grad-CAM++ channel weights (Chattopadhay et al., 2018),
                 using the usual exponential-score closed form.

    No ReLU is applied here, so callers can keep the sign.
    """
    A = activations.float()
    G = gradients.float()
    method = method.lower()
    if method == "hirescam":
        cam = (G * A).sum(dim=1)
    elif method == "gradcam":
        w = G.mean(dim=(2, 3), keepdim=True)
        cam = (w * A).sum(dim=1)
    elif method in ("gradcam++", "gradcampp", "gradcam_pp"):
        g2 = G.pow(2)
        g3 = G.pow(3)
        sum_a = A.sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * g2 + sum_a * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        w = (alpha * F.relu(G)).sum(dim=(2, 3), keepdim=True)
        cam = (w * A).sum(dim=1)
    else:
        raise ValueError(f"Unknown CAM method: {method!r} (expected one of {CAM_METHODS})")
    return cam[0]


@dataclass
class CAMResult:
    """Output of `compute_cam`."""

    output: Any                 # raw model output, detached
    logits: torch.Tensor        # [1, K], detached
    target_class: int
    method: str
    cam: np.ndarray             # signed raw map at layer resolution
    heatmap: np.ndarray         # positive evidence for target class, [0, 1]
    signed_map: np.ndarray      # signed evidence for target class, [-1, 1]


def _detach_output(output: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return output.detach()
    if isinstance(output, (tuple, list)):
        return type(output)(_detach_output(o) for o in output)
    return output


def compute_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    method: str = "hirescam",
    target_class: TargetSpec = None,
    output_size: Optional[Tuple[int, int]] = None,
    logits_fn: LogitsFn = default_logits_fn,
) -> CAMResult:
    """
    Class activation map and logits from one forward + one partial backward.

    The backward pass uses `torch.autograd.grad` with respect to the target
    layer's output only, so no parameter gradients are computed or stored.

    Args:
        model: CNN in eval mode
        input_tensor: [1, C, H, W]
        target_layer: module whose output is explained
        method: one of CAM_METHODS
        target_class: class index, a callable(logits) -> index, or None for
            the predicted class
        output_size: (H, W) of the returned maps; defaults to the input size
        logits_fn: maps the model output to [1, K] logits
    """
    if output_size is None:
        output_size = tuple(input_tensor.shape[-2:])

    with _model_lock(model), torch.enable_grad():
        # Requiring grad on the input guarantees a graph even if the
        # parameters are frozen; the input gradient itself is never computed.
        x = input_tensor.detach().clone().requires_grad_(True)
        with capture_outputs([target_layer]) as acts:
            output = model(x)
        logits = logits_fn(output)
        if not acts:
            raise RuntimeError("Target layer was not executed during the forward pass")
        activation = acts[-1]

        tc = _resolve_target(target_class, logits.detach())
        score = class_score(logits, tc)
        (grad,) = torch.autograd.grad(score, activation)

    cam = cam_from_activations(activation.detach(), grad.detach(), method)
    cam_np = cam.cpu().numpy().astype(np.float32)
    cam_up = resize_map(cam_np, output_size)

    return CAMResult(
        output=_detach_output(output),
        logits=logits.detach(),
        target_class=tc,
        method=method.lower(),
        cam=cam_np,
        heatmap=normalize_map(np.maximum(cam_up, 0.0)),
        signed_map=normalize_signed(cam_up),
    )


class GradCAM:
    """
    Backward-compatible Grad-CAM helper.

    Usage:
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam(input_tensor, target_class=1)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, method: str = "gradcam"):
        self.model = model
        self.target_layer = target_layer
        self.method = method

    def remove_hooks(self):
        """Kept for API compatibility; hooks are scoped to each call."""

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        output_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        self.model.eval()
        result = compute_cam(
            self.model, input_tensor, self.target_layer,
            method=self.method, target_class=target_class, output_size=output_size,
        )
        return result.heatmap


# ---------------------------------------------------------------------------
# Transformer attribution
# ---------------------------------------------------------------------------

def _tokens_to_grid(values: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Reshape a 1D patch-token vector to a 2D grid."""
    values = values.detach().float().cpu()
    n = values.numel()
    if grid_size is not None and grid_size[0] * grid_size[1] == n:
        return values.reshape(grid_size).numpy()
    side = int(math.sqrt(n))
    if side * side != n:
        logger.warning(f"Non-square number of patches: {n}")
        side = int(math.ceil(math.sqrt(n)))
        padded = torch.zeros(side * side)
        padded[:n] = values
        values = padded
    return values.reshape(side, side).numpy()


def _fuse_heads(attention: torch.Tensor, head_fusion: str) -> torch.Tensor:
    if attention.dim() == 2:
        return attention.unsqueeze(0)
    if attention.dim() == 3:
        return attention
    if head_fusion == "max":
        return attention.max(dim=1)[0]
    if head_fusion == "min":
        return attention.min(dim=1)[0]
    return attention.mean(dim=1)


def attention_rollout(
    attentions: Union[List[torch.Tensor], torch.Tensor],
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
    num_prefix_tokens: int = 1,
    readout_tokens: Optional[Sequence[int]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Class-agnostic attention rollout (Abnar & Zuidema, 2020).

    Args:
        attentions: per-layer attention, each [B, heads, N, N], [B, N, N] or
            [N, N]; or a stacked [layers, N, N] tensor
        discard_ratio: fraction of lowest attention weights to drop per layer
        head_fusion: "mean", "max" or "min"
        num_prefix_tokens: 1 for ViT (cls), 2 for DeiT (cls + dist)
        readout_tokens: token rows to read relevance from (default: [0])
        grid_size: patch grid (H, W); inferred as square if omitted

    Returns:
        [gh, gw] map normalised to [0, 1]
    """
    default_grid = 14
    if attentions is None:
        logger.warning("No attention tensors provided (None)")
        return np.zeros((default_grid, default_grid), dtype=np.float32)
    if isinstance(attentions, torch.Tensor):
        if attentions.numel() == 0:
            logger.warning("Empty attention tensor provided")
            return np.zeros((default_grid, default_grid), dtype=np.float32)
        attentions = [attentions[i] for i in range(attentions.shape[0])]
    if len(attentions) == 0:
        logger.warning("Empty attention list provided")
        return np.zeros((default_grid, default_grid), dtype=np.float32)

    result = None
    for attention in attentions:
        if attention.dim() not in (2, 3, 4):
            logger.warning(f"Unexpected attention shape: {attention.shape}")
            continue
        fused = _fuse_heads(attention.detach().float(), head_fusion)

        if discard_ratio > 0:
            flat = fused.reshape(fused.size(0), -1)
            threshold = torch.quantile(flat, discard_ratio, dim=1, keepdim=True).view(-1, 1, 1)
            fused = torch.where(fused < threshold, torch.zeros_like(fused), fused)
            fused = fused / (fused.sum(dim=-1, keepdim=True) + 1e-9)

        n = fused.size(-1)
        eye = torch.eye(n, device=fused.device, dtype=fused.dtype).unsqueeze(0)
        layer = 0.5 * fused + 0.5 * eye
        result = layer if result is None else torch.bmm(layer, result)

    if result is None:
        return np.zeros((default_grid, default_grid), dtype=np.float32)

    rows = list(readout_tokens) if readout_tokens else [0]
    relevance = result[0, rows, num_prefix_tokens:].mean(dim=0)
    return normalize_map(_tokens_to_grid(relevance, grid_size))


def gradient_weighted_rollout(
    attentions: Sequence[torch.Tensor],
    gradients: Sequence[torch.Tensor],
    num_prefix_tokens: int = 1,
    readout_tokens: Optional[Sequence[int]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Class-specific relevance (Chefer et al., 2021, Eq. 5-6).

    For each layer: A_bar = mean_heads(clamp(grad * attn, min=0)), then
    R <- R + A_bar @ R starting from R = I. The relevance of each patch is
    read from the readout token rows (cls for ViT, cls + dist for DeiT).

    Args:
        attentions / gradients: per-layer [1, heads, N, N]
    Returns:
        [gh, gw] map normalised to [0, 1]
    """
    n = attentions[0].shape[-1]
    device = attentions[0].device
    R = torch.eye(n, device=device, dtype=torch.float32)
    for A, G in zip(attentions, gradients):
        cam = (G.float() * A.float()).clamp(min=0).mean(dim=1)[0]  # [N, N]
        R = R + cam @ R
    rows = list(readout_tokens) if readout_tokens else [0]
    relevance = R[rows, num_prefix_tokens:].mean(dim=0)
    return normalize_map(_tokens_to_grid(relevance, grid_size))


@contextmanager
def capture_attention_probs(attn_modules: Sequence[nn.Module]) -> Iterator[List[torch.Tensor]]:
    """
    Capture the softmax attention probabilities timm computes.

    timm's Attention uses `F.scaled_dot_product_attention` when
    `fused_attn` is True, which never materialises the attention matrix.
    This temporarily switches each module to the explicit path and hooks
    `attn_drop`, whose output is exactly the matrix multiplied with V
    (q_norm/k_norm and scaling included). The flags are restored on exit.
    """
    saved = []
    hook_targets = []
    for m in attn_modules:
        if hasattr(m, "fused_attn"):
            saved.append((m, m.fused_attn))
            m.fused_attn = False
        drop = getattr(m, "attn_drop", None)
        if isinstance(drop, nn.Module):
            hook_targets.append(drop)
    try:
        with capture_outputs(hook_targets) as storage:
            yield storage
    finally:
        for m, flag in saved:
            m.fused_attn = flag


@contextmanager
def _capture_attention_qkv(attn_modules: Sequence[nn.Module]) -> Iterator[List[torch.Tensor]]:
    """Fallback: recompute attention from qkv inside a forward hook."""
    owner = threading.get_ident()
    storage: List[torch.Tensor] = []

    def hook(module, inputs, _output):
        if threading.get_ident() != owner:
            return
        x = inputs[0]
        B, N, C = x.shape
        heads = module.num_heads
        head_dim = getattr(module, "head_dim", C // heads)
        qkv = module.qkv(x).reshape(B, N, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        if hasattr(module, "q_norm"):
            q, k = module.q_norm(q), module.k_norm(k)
        scale = getattr(module, "scale", head_dim ** -0.5)
        storage.append(((q * scale) @ k.transpose(-2, -1)).softmax(dim=-1).detach())

    handles = [m.register_forward_hook(hook) for m in attn_modules]
    try:
        yield storage
    finally:
        for h in handles:
            h.remove()


@dataclass
class TransformerAttribution:
    """Output of `compute_transformer_attribution`."""

    output: Any                 # raw model output, detached
    logits: torch.Tensor        # [1, K], detached
    target_class: int
    method: str                 # method actually used
    grid: np.ndarray            # [gh, gw] in [0, 1]
    heatmap: np.ndarray         # grid resized to output_size, [0, 1]


def compute_transformer_attribution(
    model: nn.Module,
    input_tensor: torch.Tensor,
    blocks: Sequence[nn.Module],
    method: str = "grad_rollout",
    target_class: TargetSpec = None,
    num_prefix_tokens: int = 1,
    readout_tokens: Optional[Sequence[int]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    output_size: Optional[Tuple[int, int]] = None,
    logits_fn: LogitsFn = default_logits_fn,
) -> TransformerAttribution:
    """
    Logits plus an attribution map for a ViT/DeiT model, in one pass.

    Args:
        model: transformer classifier in eval mode
        input_tensor: [1, 3, H, W]
        blocks: transformer blocks, each with an `.attn` module
        method: "grad_rollout" (class-specific, default) or "rollout"
            (class-agnostic). If gradients or attention cannot be captured,
            this falls back to "rollout" (or "rollout_qkv" if timm's
            attention probabilities are unavailable).
        target_class: class index, callable(logits) -> index, or None
        num_prefix_tokens: 1 for ViT (cls), 2 for DeiT (cls + dist)
        readout_tokens: rows whose output feeds the head (default: [0])
        grid_size: patch grid (H, W)
        output_size: (H, W) of the returned heatmap; defaults to input size
    """
    if output_size is None:
        output_size = tuple(input_tensor.shape[-2:])
    method = method.lower()
    if method not in TRANSFORMER_METHODS:
        raise ValueError(f"Unknown transformer method: {method!r} (expected one of {TRANSFORMER_METHODS})")
    attn_modules = [b.attn for b in blocks if hasattr(b, "attn")]
    used = method

    with _model_lock(model):
        grad_ctx = torch.enable_grad() if method == "grad_rollout" else torch.no_grad()
        with grad_ctx, capture_attention_probs(attn_modules) as attns:
            output = model(input_tensor)
        logits = logits_fn(output)
        tc = _resolve_target(target_class, logits.detach())

        grid = None
        if len(attns) == len(attn_modules) and attns:
            if method == "grad_rollout":
                grads = None
                try:
                    score = class_score(logits, tc)
                    grads = torch.autograd.grad(score, attns, allow_unused=True)
                except RuntimeError as e:
                    logger.warning(f"Gradient rollout failed ({e}); using plain rollout")
                if grads is not None and all(g is not None for g in grads):
                    grid = gradient_weighted_rollout(
                        [a.detach() for a in attns], [g.detach() for g in grads],
                        num_prefix_tokens=num_prefix_tokens,
                        readout_tokens=readout_tokens, grid_size=grid_size,
                    )
                else:
                    used = "rollout"
            if grid is None:
                grid = attention_rollout(
                    [a.detach() for a in attns], num_prefix_tokens=num_prefix_tokens,
                    readout_tokens=readout_tokens, grid_size=grid_size,
                )
        else:
            logger.warning(
                f"Captured {len(attns)} attention maps for {len(attn_modules)} blocks; "
                "falling back to qkv-recomputed rollout"
            )
            used = "rollout_qkv"
            with torch.no_grad(), _capture_attention_qkv(attn_modules) as qkv_attns:
                model(input_tensor)
            grid = attention_rollout(
                qkv_attns, num_prefix_tokens=num_prefix_tokens,
                readout_tokens=readout_tokens, grid_size=grid_size,
            )

    return TransformerAttribution(
        output=_detach_output(output),
        logits=logits.detach(),
        target_class=tc,
        method=used,
        grid=grid,
        heatmap=normalize_map(resize_map(grid, output_size)),
    )


def compute_vit_attention_rollout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    blocks_attr: str = "blocks",
    num_prefix_tokens: int = 1,
    output_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Class-agnostic attention rollout heatmap [H, W] in [0, 1] (legacy API)."""
    model.eval()
    blocks = model
    for attr in blocks_attr.split("."):
        blocks = getattr(blocks, attr)
    return compute_transformer_attribution(
        model, input_tensor, list(blocks), method="rollout",
        num_prefix_tokens=num_prefix_tokens, output_size=output_size,
    ).heatmap


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _png_base64(rgb: np.ndarray, output_size: Optional[Tuple[int, int]] = None) -> str:
    img = Image.fromarray(rgb)
    if output_size is not None:
        img = img.resize(output_size, Image.BILINEAR)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def heatmap_to_base64(
    heatmap: np.ndarray,
    colormap: str = "turbo",
    output_size: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Encode a [0, 1] heatmap as a base64 PNG (no data: prefix).

    Args:
        heatmap: 2D array in [0, 1]
        colormap: "turbo" (default) or "diverging"
        output_size: optional (width, height)
    """
    return _png_base64(apply_colormap(heatmap, colormap), output_size)


def signed_heatmap_to_base64(
    signed_map: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Encode a [-1, 1] signed map with the diverging colormap.

    Red = evidence for the explained class, blue = evidence against it,
    white = no evidence.
    """
    values = (np.clip(np.asarray(signed_map, dtype=np.float32), -1.0, 1.0) + 1.0) / 2.0
    return _png_base64(apply_colormap(values, "diverging"), output_size)


def overlay_heatmap_on_image(
    image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "turbo",
) -> str:
    """Blend a [0, 1] heatmap over an image; returns a base64 PNG."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[:, :, :3]

    h, w = image.shape[:2]
    heatmap_resized = np.clip(resize_map(np.asarray(heatmap, dtype=np.float32), (h, w)), 0.0, 1.0)
    colored = apply_colormap(heatmap_resized, colormap)
    blended = ((1 - alpha) * image.astype(np.float32) + alpha * colored.astype(np.float32)).astype(np.uint8)
    return _png_base64(blended)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def signed_evidence_share(signed_map: np.ndarray) -> Optional[float]:
    """Share (0-1) of the absolute signed mass that is positive; None if empty."""
    s = np.asarray(signed_map, dtype=np.float32)
    pos = float(np.clip(s, 0, None).sum())
    neg = float(-np.clip(s, None, 0).sum())
    total = pos + neg
    return pos / total if total > 0 else None


def compute_focus_summary(heatmap: np.ndarray, threshold: float = 0.5) -> str:
    """
    Describe where a heatmap concentrates (spread, location, coverage).

    The description is purely geometric; it makes no claim about image
    content (faces, subjects), which the heatmap alone cannot support.
    """
    if heatmap is None or np.asarray(heatmap).size == 0:
        return "no activation data available"

    heatmap = np.nan_to_num(np.array(heatmap, dtype=np.float32))
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    h, w = heatmap.shape
    mask = heatmap > threshold
    if not mask.any():
        mask = heatmap > (heatmap.max() * 0.5)
    if not mask.any():
        return "very low activation across entire image"

    y_coords, x_coords = np.where(mask)
    centroid_y = y_coords.mean() / h
    centroid_x = x_coords.mean() / w
    spread_y = y_coords.std() / h if len(y_coords) > 1 else 0
    spread_x = x_coords.std() / w if len(x_coords) > 1 else 0
    spread = (spread_y + spread_x) / 2
    coverage = mask.sum() / mask.size

    parts = []
    if spread < 0.15:
        parts.append("highly concentrated")
    elif spread < 0.25:
        parts.append("moderately concentrated")
    else:
        parts.append("spread across")

    vertical = "upper" if centroid_y < 0.33 else "lower" if centroid_y > 0.67 else "middle"
    horizontal = "left" if centroid_x < 0.33 else "right" if centroid_x > 0.67 else "center"
    if (vertical, horizontal) == ("middle", "center"):
        location = "central region"
    else:
        location = f"{vertical}-{horizontal} region"
    parts.append(location)

    if coverage > 0.4:
        parts.append(f"(~{int(coverage * 100)}% of image)")

    summary = " ".join(parts)
    if spread > 0.3:
        summary += " (examining multiple regions)"
    return summary
