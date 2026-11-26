"""
models/cnn_baseline.py
CNN Baseline detector (EfficientNet-B0 or ResNet-50) using timm backbone.
Extracts local artifact features and provides Grad-CAM hook layer.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import timm


class CNNBaseline(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True, drop_rate: float = 0.3):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate
        )
        self.feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | Dict[str, Any]:
        feat = self.backbone(x)  # [B, feat_dim]
        logits = self.classifier(feat).squeeze(-1)  # [B]

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "cnn_feat": feat,
            "aux_cnn_logits": logits,
        }

    def get_cam_layer(self) -> nn.Module:
        """Returns the penultimate conv stage for Grad-CAM."""
        if hasattr(self.backbone, "conv_head"):
            return self.backbone.conv_head
        if hasattr(self.backbone, "layer4"):
            return self.backbone.layer4[-1]
        return list(self.backbone.children())[-2]
