"""
explain/gradcam.py
Grad-CAM implementation for CNN branch gradient attribution.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def compute(self, x: torch.Tensor, out_size: tuple[int, int] = (224, 224)) -> np.ndarray:
        """Computes Grad-CAM heatmap for a single image batch. Returns [H, W, 3] uint8."""
        self.model.eval()
        x.requires_grad_(True)

        output = self.model(x)
        logits = output["logits"] if isinstance(output, dict) else output
        score = logits.sum()
        self.model.zero_grad()
        score.backward()

        # Compute weighted activations
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = torch.relu(cam)
        cam = cam.squeeze().float().cpu().numpy()

        # Normalize and resize
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam_uint8 = (cam * 255).astype(np.uint8)

        # Upscale to output size
        pil_cam = Image.fromarray(cam_uint8, mode="L").resize(out_size, Image.BICUBIC)
        cam_resized = np.array(pil_cam, dtype=np.float32) / 255.0

        # Thermal colormap
        from forensics import _thermal_colormap
        return _thermal_colormap((cam_resized * 255).astype(np.uint8))
