"""
models/vit_baseline.py
Vision Transformer baseline detector (DINOv2 ViT-S/14 or CLIP ViT-B/16).
Extracts global contextual representations and multi-head self-attention rollout maps.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import timm


class ViTBaseline(nn.Module):
    def __init__(self, backbone_name: str = "vit_small_patch14_dinov2", pretrained: bool = True, drop_rate: float = 0.3):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            dynamic_img_size=True,
        )
        self.feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(256, 1),
        )

    def extract_patch_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts tokens:
        Returns: (cls_token [B, D], patch_tokens [B, N_patches, D])
        """
        # timm forward_features handles tokenization and transformer blocks
        tokens = self.backbone.forward_features(x)
        if hasattr(self.backbone, "global_pool") and self.backbone.global_pool == "token":
            cls_token = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
        elif tokens.ndim == 3:
            cls_token = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
        else:
            cls_token = tokens
            patch_tokens = tokens.unsqueeze(1)
        return cls_token, patch_tokens

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | Dict[str, Any]:
        cls_token, patch_tokens = self.extract_patch_features(x)
        logits = self.classifier(cls_token).squeeze(-1)

        if not return_aux:
            return logits

        # Compute CLS-to-patch attention map
        # Approximate attention from token cosine similarities
        B, N, D = patch_tokens.shape
        cls_norm = cls_token / (cls_token.norm(dim=-1, keepdim=True) + 1e-6)
        patch_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + 1e-6)
        sim = torch.bmm(patch_norm, cls_norm.unsqueeze(-1)).squeeze(-1)  # [B, N]
        grid_size = int(N ** 0.5)
        attn_map = sim.view(B, 1, grid_size, grid_size)

        return {
            "logits": logits,
            "vit_feat": cls_token,
            "aux_vit_logits": logits,
            "attention_map": attn_map,
        }
