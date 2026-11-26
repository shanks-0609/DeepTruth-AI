"""
tests/test_models.py
Unit tests verifying forward pass shapes, auxiliary outputs, and CUDA bf16 execution.
"""

import pytest
import torch
from models.registry import build_model


@pytest.mark.parametrize("model_name", ["cnn", "vit", "hybrid", "guided"])
def test_model_forward_cpu(model_name: str):
    model = build_model(model_name, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == torch.Size([2]), f"{model_name} output shape mismatch: {out.shape}"


@pytest.mark.parametrize("model_name", ["cnn", "vit", "hybrid", "guided"])
def test_model_aux_outputs(model_name: str):
    model = build_model(model_name, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    res = model(x, return_aux=True)
    assert isinstance(res, dict)
    assert "logits" in res
    assert res["logits"].shape == torch.Size([2])


def test_cuda_bf16_guided():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda:0")
    model = build_model("guided", pretrained=False).to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        res = model(x, return_aux=True)
        loss = res["logits"].sum()
        loss.backward()
    assert res["logits"].shape == torch.Size([2])
    assert model.classifier[1].weight.grad is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
