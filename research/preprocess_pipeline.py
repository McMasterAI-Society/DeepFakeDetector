# preprocess_pipeline.py
# DeepFakeDetector — Image Preprocessing Pipeline (PyTorch-focused)
# NOTE: This is a self-contained preprocessing module that prepares inputs for
# three "experts": (1) ViT global context, (2) CNN spatial patches, and
# (3) frequency-domain features (FFT on the luminance channel).
# You can import and call preprocess_all(image_path) to get ready-to-batch tensors.

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageOps

# These imports are only needed if you execute the transforms.
# It's okay if torch/torchvision aren't installed at authoring time.
try:
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
except Exception:
    torch = None
    T = None
    TF = None

# Normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLIP_MEAN     = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)

@dataclass
class PreprocessConfig:
    vit_size: int = 224            # typical for ViT-B/16; can be 384 for ViT-L/14
    cnn_patch: int = 224           # patch size for CNN 'spatial expert'
    patch_stride: float = 0.5      # stride expressed as a fraction of patch size (50% overlap)
    normalize: str = "imagenet"    # "imagenet" or "clip"
    ycbcr_for_fft: bool = True     # use luminance (Y) for FFT branch
    windowing: str = "hann"        # window to reduce spectral leakage for FFT
    preserve_exif_orientation: bool = True
    max_long_side: int = 4096      # guardrail; tile instead of aggressively downscaling

def _to_rgb(img: Image.Image) -> Image.Image:
    if isinstance(img, Image.Image):
        if cfg.preserve_exif_orientation:
            img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    raise TypeError("Expected PIL.Image")

def resize_keep_ar(img: Image.Image, long_side: int) -> Image.Image:
    """Downscale large images to a manageable size while preserving aspect ratio.
    We prefer tiling over hard downscaling to avoid destroying small artifacts."""
    w, h = img.size
    m = max(w, h)
    if m <= long_side:
        return img.copy()
    scale = long_side / float(m)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new_size, resample=Image.BICUBIC, reducing_gap=2.0)

def _norm_transform(scheme: str):
    if T is None:
        raise ImportError("torchvision is not available for transforms.")
    if scheme.lower() == "clip":
        return T.Normalize(CLIP_MEAN, CLIP_STD)
    return T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

def make_vit_tensor(img: Image.Image, size: int, scheme: str):
    if T is None:
        raise ImportError("torchvision is not available for transforms.")
    # Common ViT recipe: resize shortest side to size, then center-crop to (size, size).
    t = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop((size, size)),
        T.ToTensor(),
        _norm_transform(scheme),
    ])
    return t(img)

def extract_patches(img: Image.Image, patch: int, stride_ratio: float) -> Tuple[List[Image.Image], List[int], List[int]]:
    """Sliding-window extraction with overlap. Pads the last tiles if needed."""
    W, H = img.size
    stride = max(1, int(round(patch * stride_ratio)))
    xs = list(range(0, max(W - patch, 0) + 1, stride))
    ys = list(range(0, max(H - patch, 0) + 1, stride))
    if not xs: xs = [0]
    if not ys: ys = [0]
    patches = []
    for y in ys:
        for x in xs:
            crop = img.crop((x, y, x + patch, y + patch))
            if crop.size != (patch, patch):
                pad = Image.new("RGB", (patch, patch))
                pad.paste(crop, (0, 0))
                crop = pad
            patches.append(crop)
    return patches, xs, ys

def ycbcr_channels(img: Image.Image):
    ycbcr = img.convert("YCbCr")
    Y, Cb, Cr = ycbcr.split()
    return Y, Cb, Cr

def _window(arr: np.ndarray, kind: str):
    h, w = arr.shape[:2]
    if kind == "hann":
        wy = np.hanning(h)[:, None]
        wx = np.hanning(w)[None, :]
        return wy * wx
    if kind == "hamming":
        wy = np.hamming(h)[:, None]
        wx = np.hamming(w)[None, :]
        return wy * wx
    return np.ones((h, w), dtype=arr.dtype)

def fft_log_magnitude(gray_img: Image.Image, window: str = "hann") -> Image.Image:
    """Compute log-magnitude spectrum (shifted) as a single-channel image in [0, 255]."""
    arr = np.array(gray_img).astype(np.float32) / 255.0
    win = _window(arr, window)
    F = np.fft.fft2(arr * win)
    Fshift = np.fft.fftshift(F)
    mag = np.log1p(np.abs(Fshift))
    # standardize then min-max to [0, 1]
    mag = (mag - mag.mean()) / (mag.std() + 1e-6)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    return Image.fromarray((mag * 255).astype(np.uint8))

def patches_to_tensor(patches: List[Image.Image], scheme: str):
    if torch is None or T is None:
        raise ImportError("torch/torchvision not available for tensor conversion.")
    t = T.Compose([
        T.ToTensor(),
        _norm_transform(scheme),
    ])
    return torch.stack([t(p) for p in patches], dim=0)

def preprocess_all(path: str, cfg: PreprocessConfig = PreprocessConfig()) -> Dict:
    """Full pipeline producing inputs for each expert and a bit of metadata.
    Returns a dict with 'vit', 'cnn_patches', 'fft_tensor', 'meta' keys."""
    img = Image.open(path)
    if cfg.preserve_exif_orientation:
        img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    # Prefer tiling to preserve artifacts in very large images
    img = resize_keep_ar(img, cfg.max_long_side)

    # Global ViT input
    vit_tensor = make_vit_tensor(img, cfg.vit_size, "imagenet")

    # CNN spatial patches
    patches, xs, ys = extract_patches(img, cfg.cnn_patch, cfg.patch_stride)
    cnn_patches = patches_to_tensor(patches, "imagenet")

    # Frequency-domain: FFT of luminance; shape (1, H, W)
    Y, Cb, Cr = ycbcr_channels(img.resize((cfg.cnn_patch, cfg.cnn_patch), Image.BICUBIC))
    fft_img = fft_log_magnitude(Y, cfg.windowing)
    if T is None:
        raise ImportError("torchvision is required to turn FFT image into a tensor.")
    fft_tensor = T.ToTensor()(fft_img)  # [1, H, W], already normalized to [0,1]

    return {
        "vit": vit_tensor,              # torch.Tensor [3, S, S]
        "cnn_patches": cnn_patches,     # torch.Tensor [N, 3, P, P]
        "fft_tensor": fft_tensor,       # torch.Tensor [1, P, P]
        "meta": {
            "grid_x": xs,
            "grid_y": ys,
            "patch_size": cfg.cnn_patch,
            "stride_ratio": cfg.patch_stride,
        }
    }

if __name__ == "__main__":
    print("This module prepares inputs for ViT/CNN/FFT experts.")
    print("Example usage:")
    print("  from preprocess_pipeline import preprocess_all, PreprocessConfig")
    print("  batch = preprocess_all('example.jpg', PreprocessConfig())")
    print("  print(batch['vit'].shape, batch['cnn_patches'].shape, batch['fft_tensor'].shape)")
