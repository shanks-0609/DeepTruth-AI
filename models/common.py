"""
models/common.py
Common neural components:
- FiLM (Feature-wise Linear Modulation)
- Spatial Attention Gating
- Cross-Attention Fusion
- SRM (Spatial Rich Model) High-Pass Filter Bank
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.
    Predicts channel-wise scale (gamma) and shift (beta) from conditioning vector (e.g. ViT CLS).
    """

    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * num_features)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: [B, cond_dim], x: [B, C, H, W]
        film_params = self.proj(cond)  # [B, 2*C]
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta


class SpatialGuidanceGating(nn.Module):
    """
    Gating module that projects a 1-channel spatial attention map A to the CNN stage dimension.
    F' = F * (1.0 + sigmoid(conv(A_upsampled)))
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, attn_map: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W], attn_map: [B, 1, H_p, W_p]
        if attn_map.shape[-2:] != feat.shape[-2:]:
            attn_upsampled = F.interpolate(
                attn_map, size=feat.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            attn_upsampled = attn_map
        gate = self.conv(attn_upsampled)
        return feat * (1.0 + gate)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention block where CNN features attend to ViT patch tokens.
    Queries: CNN tokens [B, N_cnn, D]
    Keys/Values: ViT patch tokens [B, N_vit, D]
    """

    def __init__(self, cnn_dim: int, vit_dim: int, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(cnn_dim, embed_dim)
        self.k_proj = nn.Linear(vit_dim, embed_dim)
        self.v_proj = nn.Linear(vit_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, cnn_dim)
        self.norm = nn.LayerNorm(cnn_dim)

    def forward(self, cnn_tokens: torch.Tensor, vit_tokens: torch.Tensor) -> torch.Tensor:
        # cnn_tokens: [B, N_c, cnn_dim], vit_tokens: [B, N_v, vit_dim]
        q = self.q_proj(cnn_tokens)
        k = self.k_proj(vit_tokens)
        v = self.v_proj(vit_tokens)
        attn_out, _ = self.mha(q, k, v)
        out = self.norm(cnn_tokens + self.out_proj(attn_out))
        return out


class SRMHighPassFilter(nn.Module):
    """
    Fixed Spatial Rich Model (SRM) high-pass filter bank for micro-noise residual extraction.
    Filters: 3 basic kernels commonly used in image forensics (KB, KV, H).
    """

    def __init__(self):
        super().__init__()
        # 3 fixed 5x5 filters
        f1 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32) / 4.0

        f2 = torch.tensor([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ], dtype=torch.float32) / 12.0

        f3 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, -4, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32)

        filters = torch.stack([f1, f2, f3]).unsqueeze(1)  # [3, 1, 5, 5]
        self.register_buffer("weights", filters)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        # Convert RGB to Luminance: [B, 1, H, W]
        r, g, b = x_rgb[:, 0:1], x_rgb[:, 1:2], x_rgb[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return F.conv2d(gray, self.weights, padding=2)
