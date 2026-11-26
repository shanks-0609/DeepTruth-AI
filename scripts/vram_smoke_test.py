"""
scripts/vram_smoke_test.py
Runs one forward+backward pass for each model under bf16 autocast
with the target micro-batch size to verify VRAM fits < 7.5 GB.
"""

import torch
import torch.nn as nn
from models.registry import build_model

MICRO_BATCH = 16
CROP = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

MODELS = [
    ("cnn",    {"backbone_name": "efficientnet_b0", "pretrained": False}),
    ("vit",    {"backbone_name": "vit_small_patch14_dinov2", "pretrained": False}),
    ("hybrid", {"cnn_name": "efficientnet_b0", "vit_name": "vit_small_patch14_dinov2", "pretrained": False}),
    ("guided", {"cnn_name": "efficientnet_b0", "vit_name": "vit_small_patch14_dinov2",
                "pretrained": False, "use_freq_stream": True, "use_spatial_gate": True}),
]

print("=" * 60)
print(f"VRAM SMOKE TEST  device={device}  dtype={dtype}  micro_batch={MICRO_BATCH}")
print("=" * 60)
for name, kwargs in MODELS:
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    model = build_model(name, **kwargs).to(device)
    model = model.to(memory_format=torch.channels_last)
    x = torch.randn(MICRO_BATCH, 3, CROP, CROP, device=device, dtype=torch.float32)
    x = x.to(memory_format=torch.channels_last)
    labels = torch.randint(0, 2, (MICRO_BATCH,), device=device, dtype=torch.float32)

    return_aux = hasattr(model, "cross_attn") or hasattr(model, "aux_cnn_head")
    with torch.autocast(device_type="cuda", dtype=dtype):
        out = model(x, return_aux=return_aux)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    gate_ok = peak_mb < 7500
    status = "PASS ✓" if gate_ok else "FAIL ✗ EXCEEDS 7.5GB"
    print(f"  {name:8s}: peak VRAM = {peak_mb:6.0f} MB  [{status}]")

    del model, x, labels, out, loss
    torch.cuda.empty_cache()

print("=" * 60)
