"""
models/guided_net.py
ViT-Guided CNN Architecture:
- Global branch: ViT-S/14 (DINOv2) extracts semantic context, CLS token & attention map.
- Local branch: EfficientNet-B0 (or ResNet-50) extracts micro-pixel artifact features.
- Guidance: ViT attention rollout/importance map spatial-gates CNN stage 2/3,
  and ViT CLS token modulates CNN channels via FiLM.
- Fusion: CNN feature tokens cross-attend to ViT patch tokens, concatenated with CLS token.
- Returns main logits, branch auxiliary logits, attention maps, and CAM hook.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.common import FiLM, SpatialGuidanceGating, CrossAttentionFusion, SRMHighPassFilter


class TinyFreqNet(nn.Module):
    """Tiny CNN processing SRM high-pass residual features."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.srm = SRMHighPassFilter()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = self.srm(x)
        return self.net(residuals)


class ViTGuidedCNN(nn.Module):
    def __init__(
        self,
        cnn_name: str = "efficientnet_b0",
        vit_name: str = "vit_small_patch14_dinov2",
        pretrained: bool = True,
        use_freq_stream: bool = True,
        use_spatial_gate: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.cnn_name = cnn_name
        self.vit_name = vit_name
        self.use_freq_stream = use_freq_stream
        self.use_spatial_gate = use_spatial_gate

        # 1. ViT Backbone
        self.vit = timm.create_model(
            vit_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            dynamic_img_size=True,
        )
        self.vit_dim = self.vit.num_features  # 384 for vit_small

        # 2. CNN Backbone
        self.cnn = timm.create_model(
            cnn_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate
        )
        self.cnn_dim = self.cnn.num_features  # 1280 for efficientnet_b0

        # CNN intermediate channel dimensions for EfficientNet-B0
        self.stage2_dim = 40

        # 3. Guidance Mechanisms
        if self.use_spatial_gate:
            self.spatial_gate = SpatialGuidanceGating(in_channels=self.stage2_dim)
        self.film = FiLM(cond_dim=self.vit_dim, num_features=self.stage2_dim)

        # 4. Cross Attention & Fusion
        self.cross_attn = CrossAttentionFusion(
            cnn_dim=self.cnn_dim, vit_dim=self.vit_dim, embed_dim=256, num_heads=4
        )

        # Optional Frequency Stream
        self.freq_dim = 64 if self.use_freq_stream else 0
        if self.use_freq_stream:
            self.freq_net = TinyFreqNet(out_dim=self.freq_dim)

        # 5. Classification Heads
        fused_total_dim = self.cnn_dim + self.vit_dim + self.freq_dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(fused_total_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(256, 1),
        )

        # Auxiliary Branch Heads (for independent branch evaluation)
        self.aux_cnn_head = nn.Linear(self.cnn_dim, 1)
        self.aux_vit_head = nn.Linear(self.vit_dim, 1)

    def extract_vit_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts ViT CLS token, patch tokens, and spatial importance attention map.
        """
        tokens = self.vit.forward_features(x)
        if hasattr(self.vit, "global_pool") and self.vit.global_pool == "token":
            cls_token = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
        elif tokens.ndim == 3:
            cls_token = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
        else:
            cls_token = tokens
            patch_tokens = tokens.unsqueeze(1)

        B, N, D = patch_tokens.shape
        cls_norm = cls_token / (cls_token.norm(dim=-1, keepdim=True) + 1e-6)
        patch_norm = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + 1e-6)
        sim = torch.bmm(patch_norm, cls_norm.unsqueeze(-1)).squeeze(-1)  # [B, N]
        grid_size = int(N ** 0.5)
        attn_map = sim.view(B, 1, grid_size, grid_size)

        return cls_token, patch_tokens, attn_map

    def forward_cnn_guided(
        self, x: torch.Tensor, attn_map: torch.Tensor, cls_token: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes CNN stages with spatial gating and FiLM modulation.
        """
        if hasattr(self.cnn, "conv_stem"):
            # EfficientNet structure
            h = self.cnn.conv_stem(x)
            h = self.cnn.bn1(h)
            # Blocks 0 to 2
            for block in self.cnn.blocks[:3]:
                h = block(h)

            # Inject ViT Spatial Guidance & FiLM modulation
            if self.use_spatial_gate:
                h = self.spatial_gate(h, attn_map)
            h = self.film(h, cls_token)

            # Remaining blocks 3 to end
            for block in self.cnn.blocks[3:]:
                h = block(h)

            h = self.cnn.conv_head(h)
            feat_map = self.cnn.bn2(h)  # [B, 1280, 7, 7]
            pooled = self.cnn.global_pool(feat_map)
        else:
            # Fallback for standard ResNet
            feat_map = self.cnn.forward_features(x)
            pooled = self.cnn.forward_head(feat_map, pre_logits=True)

        return pooled, feat_map

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | Dict[str, Any]:
        # 1. ViT global representation
        cls_token, patch_tokens, attn_map = self.extract_vit_features(x)

        # 2. Guided CNN forward
        pooled_cnn, feat_map = self.forward_cnn_guided(x, attn_map, cls_token)

        # 3. Cross-Attention
        B, C, H, W = feat_map.shape
        cnn_tokens = feat_map.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        cnn_attended = self.cross_attn(cnn_tokens, patch_tokens)
        cnn_fused = cnn_attended.mean(dim=1)  # [B, C]

        # 4. Optional Frequency stream
        fused_components = [cnn_fused, cls_token]
        if self.use_freq_stream:
            freq_feat = self.freq_net(x)
            fused_components.append(freq_feat)

        z_total = torch.cat(fused_components, dim=-1)
        main_logits = self.classifier(z_total).squeeze(-1)

        if not return_aux:
            return main_logits

        aux_cnn_logits = self.aux_cnn_head(pooled_cnn).squeeze(-1)
        aux_vit_logits = self.aux_vit_head(cls_token).squeeze(-1)

        return {
            "logits": main_logits,
            "aux_cnn_logits": aux_cnn_logits,
            "aux_vit_logits": aux_vit_logits,
            "attention_map": attn_map,
            "cls_token": cls_token,
            "pooled_cnn": pooled_cnn,
            "fused_feat": z_total,
        }

    def get_cam_layer(self) -> nn.Module:
        if hasattr(self.cnn, "conv_head"):
            return self.cnn.conv_head
        return list(self.cnn.children())[-2]
