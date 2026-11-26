"""
explain/attention.py
Attention map extraction for ViT branch (CLS-to-patch similarity rollout).
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def extract_attention_map(vit_model, x: torch.Tensor, out_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Extracts CLS-to-patch cosine attention map from a timm ViT model.
    Returns [H, W, 3] uint8 thermal heatmap.
    """
    with torch.no_grad():
        tokens = vit_model.forward_features(x)
        if tokens.ndim == 3:
            cls_token = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
        else:
            return np.zeros((*out_size, 3), dtype=np.uint8)

    B, N, D = patch_tokens.shape
    cls_norm = cls_token / (cls_token.norm(dim=-1, keepdim=True) + 1e-6)
    patch_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + 1e-6)
    sim = torch.bmm(patch_norm, cls_norm.unsqueeze(-1)).squeeze(-1)  # [B, N]

    # Take first batch, reshape to grid
    sim_np = sim[0].float().cpu().numpy()
    grid_size = int(N ** 0.5)
    attn_grid = sim_np.reshape(grid_size, grid_size)

    # Normalize
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-6)
    attn_uint8 = (attn_grid * 255).astype(np.uint8)
    pil_attn = Image.fromarray(attn_uint8, mode="L").resize(out_size, Image.BICUBIC)
    attn_resized = np.array(pil_attn, dtype=np.uint8)

    from forensics import _thermal_colormap
    return _thermal_colormap(attn_resized)
