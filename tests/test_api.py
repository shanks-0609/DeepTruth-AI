"""
tests/test_api.py
Tests verifying that FastAPI endpoints maintain complete backward-compatibility
with existing UI and contract while returning new additive forensic fields.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io
import numpy as np


@pytest.fixture
def client():
    return TestClient(app)


def test_api_info(client):
    r = client.get("/api/info")
    assert r.status_code == 200
    data = r.json()
    assert "title" in data
    assert "team" in data
    assert "models_evaluated" in data


def test_api_samples(client):
    r = client.get("/api/samples")
    assert r.status_code == 200
    samples = r.json()
    assert len(samples) >= 4
    for s in samples:
        assert "id" in s
        assert "url" in s
        assert "category" in s


def test_api_benchmark(client):
    r = client.get("/api/benchmark")
    assert r.status_code == 200
    data = r.json()
    assert "dataset_summary" in data or "metrics" in data


def test_api_analyze_contract_backward_compatible(client):
    """Verifies that all legacy keys expected by static/js/app.js are present."""
    r = client.post("/api/analyze", data={"sample_id": "midjourney_astronaut"})
    assert r.status_code == 200
    data = r.json()

    # Legacy fields REQUIRED by existing UI
    expected_legacy_keys = [
        "verdict",
        "is_fake",
        "confidence",
        "generator_type",
        "dimensions",
        "reasoning",
        "models",
        "heatmaps",
        "forensics",
    ]
    for k in expected_legacy_keys:
        assert k in data, f"Missing legacy key: {k}"

    assert "cnn" in data["models"]
    assert "vit" in data["models"]
    assert "hybrid" in data["models"]

    assert "ela" in data["heatmaps"]
    assert "vit_attention" in data["heatmaps"]

    assert "noise_uniformity" in data["forensics"]
    assert "patch_discrepancy" in data["forensics"]
    assert "spectral_frequency_energy" in data["forensics"]
    assert "laplacian_gradient_variance" in data["forensics"]


def test_api_analyze_custom_upload(client):
    # Create synthetic test image
    img = Image.new("RGB", (256, 256), color=(128, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    r = client.post(
        "/api/analyze",
        files={"file": ("test_synthetic.jpg", buf, "image/jpeg")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "verdict" in data
    assert "confidence" in data
    assert "domain_warning" in data


if __name__ == "__main__":
    pytest.main(["-v", __file__])
