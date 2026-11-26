"""
models/concat_hybrid.py
Standard late feature concatenation network:
CNN (EfficientNet-B0) + ViT (ViT-S/14 DINOv2)
Fused embedding: z_fused = [z_CNN || z_ViT] -> MLP classifier head.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from models.cnn_baseline import CNNBaseline
from models.vit_baseline import ViTBaseline


class ConcatHybridNet(nn.Module):
    def __init__(
        self,
        cnn_name: str = "efficientnet_b0",
        vit_name: str = "vit_small_patch14_dinov2",
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.cnn_branch = CNNBaseline(backbone_name=cnn_name, pretrained=pretrained, drop_rate=drop_rate)
        self.vit_branch = ViTBaseline(backbone_name=vit_name, pretrained=pretrained, drop_rate=drop_rate)

        fused_dim = self.cnn_branch.feat_dim + self.vit_branch.feat_dim

        self.fusion_head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | Dict[str, Any]:
        cnn_out = self.cnn_branch(x, return_aux=True)
        vit_out = self.vit_branch(x, return_aux=True)

        z_cnn = cnn_out["cnn_feat"]
        z_vit = vit_out["vit_feat"]

        z_fused = torch.cat([z_cnn, z_vit], dim=-1)
        logits = self.fusion_head(z_fused).squeeze(-1)

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "aux_cnn_logits": cnn_out["logits"],
            "aux_vit_logits": vit_out["logits"],
            "fused_feat": z_fused,
            "attention_map": vit_out.get("attention_map"),
        }
