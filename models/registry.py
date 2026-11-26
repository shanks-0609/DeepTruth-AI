"""
models/registry.py
Model registry providing factory method build_model for all 4 architectures:
- 'cnn': CNNBaseline (EfficientNet-B0)
- 'vit': ViTBaseline (ViT-S/14 DINOv2)
- 'hybrid': ConcatHybridNet (Late concatenation)
- 'guided': ViTGuidedCNN (Proposed ViT-guided CNN)
"""

from typing import Dict, Any, Type
import torch.nn as nn
from models.cnn_baseline import CNNBaseline
from models.vit_baseline import ViTBaseline
from models.concat_hybrid import ConcatHybridNet
from models.guided_net import ViTGuidedCNN

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "cnn": CNNBaseline,
    "vit": ViTBaseline,
    "hybrid": ConcatHybridNet,
    "guided": ViTGuidedCNN,
}


def build_model(name: str, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](**kwargs)
