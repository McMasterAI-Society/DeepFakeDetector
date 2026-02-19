"""
Explainability utilities for DeepFake detection models.

Provides:
- GradCAM: For CNN-based models (EfficientNet, CompactGradientNet)
- Attention Rollout: For ViT/DeiT transformer models
- Heatmap visualization utilities
"""

import base64
import io
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN models.
    
    Computes importance heatmaps by weighting feature map activations
    by the gradients flowing into them from the target class.
    
    Usage:
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam(input_tensor, target_class=1)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: The CNN model
            target_layer: The convolutional layer to compute Grad-CAM on
                         (typically the last conv layer before pooling)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks: List = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        output_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Class index to compute gradients for.
                         If None, uses the predicted class.
            output_size: Size to resize the heatmap to (H, W)
        
        Returns:
            Normalized heatmap as numpy array [H, W] in range [0, 1]
        """
        self.model.eval()
        
        # Enable gradients for this forward pass
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle different output formats
        if isinstance(output, tuple):
            logits = output[0]  # Some models return (logits, embeddings)
        else:
            logits = output
        
        # Ensure logits is 2D [batch, classes]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        if logits.shape[-1] > 1:
            # Multi-class: select target class score
            target_score = logits[0, target_class]
        else:
            # Binary with single output: use the logit directly
            target_score = logits[0, 0] if target_class == 1 else -logits[0, 0]
        
        target_score.backward(retain_graph=True)
        
        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            logger.warning("Gradients or activations not captured")
            return np.zeros(output_size, dtype=np.float32)
        
        # Global average pool gradients to get weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        
        # Resize to output size
        cam = F.interpolate(
            cam,
            size=output_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Convert to numpy
        heatmap = cam.squeeze().cpu().numpy()
        
        return heatmap
    
    def __del__(self):
        self.remove_hooks()


def attention_rollout(
    attentions: Union[List[torch.Tensor], torch.Tensor],
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
    num_prefix_tokens: int = 1
) -> np.ndarray:
    """
    Compute attention rollout for Vision Transformers.
    
    Aggregates attention across all layers by matrix multiplication,
    accounting for residual connections.
    
    Args:
        attentions: Attention tensors from each layer. Can be:
                   - List of tensors, each shape [batch, num_heads, seq_len, seq_len] or [seq_len, seq_len]
                   - Stacked tensor of shape [num_layers, seq_len, seq_len] (already head-fused)
        discard_ratio: Fraction of lowest attention weights to discard
        head_fusion: How to combine attention heads ("mean", "max", "min")
        num_prefix_tokens: Number of special tokens (1 for ViT cls, 2 for DeiT cls+dist)
    
    Returns:
        Attention map as numpy array of shape (grid_size, grid_size)
    """
    # Default grid size for ViT-Base (14x14 patches from 224x224 with 16x16 patch size)
    default_grid_size = 14
    
    # Handle empty input
    if attentions is None:
        logger.warning("No attention tensors provided (None)")
        return np.zeros((default_grid_size, default_grid_size), dtype=np.float32)
    
    # Convert tensor to list if needed
    if isinstance(attentions, torch.Tensor):
        if attentions.numel() == 0:
            logger.warning("Empty attention tensor provided")
            return np.zeros((default_grid_size, default_grid_size), dtype=np.float32)
        # Convert stacked tensor to list
        attentions = [attentions[i] for i in range(attentions.shape[0])]
    
    # Check if list is empty
    if len(attentions) == 0:
        logger.warning("Empty attention list provided")
        return np.zeros((default_grid_size, default_grid_size), dtype=np.float32)
    
    result = None
    
    for attention in attentions:
        # Handle different input formats
        if attention.dim() == 2:
            # Already fused: [seq_len, seq_len]
            attention_fused = attention.unsqueeze(0)  # [1, seq, seq]
        elif attention.dim() == 3:
            # Batched without heads or already fused: [B, seq, seq]
            attention_fused = attention
        elif attention.dim() == 4:
            # Full attention: [B, heads, seq, seq] - fuse heads
            if head_fusion == "mean":
                attention_fused = attention.mean(dim=1)  # [B, seq, seq]
            elif head_fusion == "max":
                attention_fused = attention.max(dim=1)[0]
            elif head_fusion == "min":
                attention_fused = attention.min(dim=1)[0]
            else:
                attention_fused = attention.mean(dim=1)
        else:
            logger.warning(f"Unexpected attention shape: {attention.shape}")
            continue
        
        # Discard low attention (optional)
        if discard_ratio > 0:
            flat = attention_fused.view(attention_fused.size(0), -1)
            threshold = torch.quantile(flat, discard_ratio, dim=1, keepdim=True)
            threshold = threshold.view(attention_fused.size(0), 1, 1)
            attention_fused = torch.where(
                attention_fused < threshold,
                torch.zeros_like(attention_fused),
                attention_fused
            )
            # Renormalize
            attention_fused = attention_fused / (attention_fused.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Add identity for residual connection
        seq_len = attention_fused.size(-1)
        identity = torch.eye(seq_len, device=attention_fused.device, dtype=attention_fused.dtype)
        attention_with_residual = 0.5 * attention_fused + 0.5 * identity.unsqueeze(0)
        
        # Matrix multiply through layers
        if result is None:
            result = attention_with_residual
        else:
            result = torch.bmm(attention_with_residual, result)
    
    # Extract CLS token attention to all patch tokens
    # result shape: [B, seq_len, seq_len]
    # CLS token is at index 0, patches start at index num_prefix_tokens
    cls_attention = result[0, 0, num_prefix_tokens:]  # [num_patches]
    
    # Reshape to grid
    num_patches = cls_attention.size(0)
    grid_size = int(math.sqrt(num_patches))
    
    if grid_size * grid_size != num_patches:
        logger.warning(f"Non-square number of patches: {num_patches}")
        # Pad or truncate to nearest square
        grid_size = int(math.ceil(math.sqrt(num_patches)))
        padded = torch.zeros(grid_size * grid_size, device=cls_attention.device)
        padded[:num_patches] = cls_attention
        cls_attention = padded
    
    attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
    
    # Normalize
    attention_map = attention_map - attention_map.min()
    if attention_map.max() > 0:
        attention_map = attention_map / attention_map.max()
    
    return attention_map


def heatmap_to_base64(
    heatmap: np.ndarray,
    colormap: str = "turbo",
    output_size: Optional[Tuple[int, int]] = None
) -> str:
    """
    Convert a heatmap array to base64-encoded PNG string.
    
    Args:
        heatmap: 2D numpy array with values in [0, 1]
        colormap: Matplotlib colormap name ("turbo", "jet", "viridis", "inferno")
        output_size: Optional (width, height) to resize to
    
    Returns:
        Base64-encoded PNG string (without data:image/png;base64, prefix)
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Apply colormap (returns RGBA)
    colored = cmap(heatmap)
    
    # Convert to uint8 RGB
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # Create PIL image
    img = Image.fromarray(rgb)
    
    # Resize if needed
    if output_size is not None:
        img = img.resize(output_size, Image.BILINEAR)
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    
    # Encode to base64
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return encoded


def overlay_heatmap_on_image(
    image: Union[np.ndarray, Image.Image],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "turbo"
) -> str:
    """
    Overlay a heatmap on an image and return as base64 PNG.
    
    Args:
        image: Original image (numpy array HWC or PIL Image)
        heatmap: 2D heatmap array [0, 1]
        alpha: Blend factor (0 = image only, 1 = heatmap only)
        colormap: Matplotlib colormap name
    
    Returns:
        Base64-encoded PNG of the overlaid image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    
    # Convert image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is uint8 RGB
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[:, :, :3]
    
    # Resize heatmap to match image size
    h, w = image.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend
    blended = (
        (1 - alpha) * image.astype(np.float32) +
        alpha * heatmap_colored.astype(np.float32)
    ).astype(np.uint8)
    
    # Convert to base64
    img = Image.fromarray(blended)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class AttentionExtractor:
    """
    Hook-based attention extractor for ViT/DeiT models.
    
    Registers hooks on transformer blocks to capture attention weights
    during forward pass.
    
    Usage:
        extractor = AttentionExtractor(model.blocks)
        output = model(input)
        attentions = extractor.get_attentions()
        extractor.clear()
    """
    
    def __init__(self, blocks: nn.ModuleList):
        """
        Args:
            blocks: List of transformer blocks (each should have .attn attribute)
        """
        self.attentions: List[torch.Tensor] = []
        self._hooks: List = []
        
        for block in blocks:
            if hasattr(block, 'attn'):
                # Hook into the attention module
                # We need to capture after softmax, before dropout
                # timm stores attention in attn.attn_drop or we can compute from qkv
                hook = block.attn.register_forward_hook(self._make_hook())
                self._hooks.append(hook)
    
    def _make_hook(self):
        """Create a forward hook that captures attention weights."""
        def hook(module, input, output):
            # For timm ViT, we need to recompute attention from qkv
            # The module receives x and outputs x after attention
            # We'll store a flag and compute in get_attentions
            pass
        return hook
    
    def extract_attention_from_block(
        self,
        block: nn.Module,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract attention weights from a single transformer block.
        
        Args:
            block: Transformer block with attention module
            x: Input tensor [B, seq_len, embed_dim]
        
        Returns:
            Attention weights [B, num_heads, seq_len, seq_len]
        """
        attn = block.attn
        B, N, C = x.shape
        
        # Get qkv
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        scale = (C // attn.num_heads) ** -0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.softmax(dim=-1)
        
        return attn_weights
    
    def get_attentions(self) -> List[torch.Tensor]:
        """Return captured attention tensors."""
        return self.attentions
    
    def clear(self):
        """Clear captured attentions."""
        self.attentions.clear()
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __del__(self):
        self.remove_hooks()


def compute_vit_attention_rollout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    blocks_attr: str = "blocks",
    num_prefix_tokens: int = 1,
    output_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Compute attention rollout for a ViT-style model.
    
    Args:
        model: The ViT model (should have .blocks attribute with transformer layers)
        input_tensor: Input image tensor [1, 3, H, W]
        blocks_attr: Attribute name for transformer blocks (e.g., "blocks" or "vit.blocks")
        num_prefix_tokens: Number of prefix tokens (1 for CLS, 2 for CLS+DIST)
        output_size: Size to resize output heatmap
    
    Returns:
        Attention heatmap as numpy array [H, W] in range [0, 1]
    """
    model.eval()
    
    # Navigate to blocks
    blocks = model
    for attr in blocks_attr.split('.'):
        blocks = getattr(blocks, attr)
    
    attentions = []
    
    # Hook to capture attention weights
    def make_attn_hook(storage):
        def hook(module, input, output):
            # Recompute attention weights
            x = input[0]
            B, N, C = x.shape
            
            # Get qkv projection
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention weights
            scale = (C // module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            
            storage.append(attn.detach())
        return hook
    
    # Register hooks
    hooks = []
    for block in blocks:
        if hasattr(block, 'attn'):
            h = block.attn.register_forward_hook(make_attn_hook(attentions))
            hooks.append(h)
    
    try:
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Compute rollout
        if attentions:
            rollout = attention_rollout(
                attentions,
                num_prefix_tokens=num_prefix_tokens
            )
            
            # Resize to output size
            rollout_img = Image.fromarray((rollout * 255).astype(np.uint8))
            rollout_img = rollout_img.resize(output_size, Image.BILINEAR)
            rollout = np.array(rollout_img) / 255.0
            
            return rollout
        else:
            logger.warning("No attention weights captured")
            return np.zeros(output_size, dtype=np.float32)
    
    finally:
        # Clean up hooks
        for h in hooks:
            h.remove()
