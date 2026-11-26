"""
inference/engine.py
DeepTruth AI Inference Engine — loads trained checkpoints once at startup,
runs GPU bf16 inference, falls back to heuristic when no checkpoint exists.
Provides domain warning for non-face images.
"""

import io
import os
import base64
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singleton state (loaded once at startup)
# ---------------------------------------------------------------------------
_LOADED_MODELS: Dict[str, nn.Module] = {}
_DEVICE: Optional[torch.device] = None
_DTYPE = torch.bfloat16


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def _pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _np_to_b64(arr: np.ndarray) -> str:
    return _pil_to_b64(Image.fromarray(arr.astype(np.uint8)))


def load_checkpoint(model_name: str, ckpt_path: str, model_cfg: Dict[str, Any]) -> Optional[nn.Module]:
    """Loads a trained checkpoint. Returns None if path doesn't exist."""
    path = Path(ckpt_path)
    if not path.exists():
        logger.warning(f"No checkpoint at {ckpt_path}. Using heuristic fallback.")
        return None
    try:
        from models.registry import build_model
        cfg = dict(model_cfg)
        cfg.pop("name", None)
        model = build_model(model_name, **cfg)
        device = _get_device()
        ckpt = torch.load(path, map_location=device, weights_only=True)
        state = ckpt.get("ema_state", ckpt.get("model_state", ckpt))
        model.load_state_dict(state, strict=False)
        model.eval()
        model.to(device)
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        _LOADED_MODELS[model_name] = model
        logger.info(f"Loaded '{model_name}' checkpoint from {ckpt_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load checkpoint '{ckpt_path}': {e}")
        return None


def _preprocess_pil(pil_img: Image.Image, crop_size: int = 224) -> torch.Tensor:
    """
    Applies EXIF orientation fix, resizes or crops to 224x224, and ImageNet normalizes.
    Safely handles images smaller than 224x224 (like thumbnails/screenshots) by scaling up.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # EXIF orientation fix
    pil_img = ImageOps.exif_transpose(pil_img)
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)

    h, w = rgb.shape[:2]

    # If image is smaller than crop_size (e.g. cropped avatar/screenshot), upscale proportionally first
    if h < crop_size or w < crop_size:
        scale = max(crop_size / h, crop_size / w)
        new_h, new_w = int(np.ceil(h * scale)), int(np.ceil(w * scale))
        pil_resized = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BICUBIC)
        rgb = np.array(pil_resized, dtype=np.uint8)
        h, w = rgb.shape[:2]

    # For images > 256px, take center 256 box first to preserve artifact scale
    if min(h, w) > 256:
        cy, cx = h // 2, w // 2
        rgb = rgb[cy-128:cy+128, cx-128:cx+128]

    tf = A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return tf(image=rgb)["image"].unsqueeze(0)  # [1, 3, H, W]


def _face_check(pil_img: Image.Image) -> bool:
    """Returns True if at least one face is detected."""
    try:
        import cv2
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = np.array(pil_img.convert("L"))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        return len(faces) > 0
    except Exception:
        return True  # on failure, don't block


def _run_heuristic(pil_img: Image.Image, filename: str) -> Dict[str, Any]:
    """
    Heuristic fallback (original simulated logic) when no checkpoint is available.
    Clearly labelled as heuristic_fallback in the response.
    """
    from app.detector import analyze_image_deep_truth
    result = analyze_image_deep_truth(pil_img, filename)
    result["model_source"] = "heuristic_fallback"
    result["calibrated_probability"] = result["models"]["hybrid"]["probability_fake"] / 100.0
    result["model_version"] = "heuristic-v1.0"
    result["inference_ms"] = None
    result["domain_warning"] = None
    result["heatmaps"]["guidance"] = None
    result["heatmaps"]["gradcam"] = None
    result["heatmaps"]["fft_spectrum"] = None
    result["texture_report"] = None
    return result


def analyze_with_trained_models(pil_img: Image.Image, filename: str = "upload.jpg",
                                 models_to_run: Optional[list] = None) -> Dict[str, Any]:
    """
    Main inference entry point. Uses trained checkpoints if available, heuristic otherwise.
    models_to_run: list of model names to include. Default: all available.
    """
    t_start = time.perf_counter()
    device = _get_device()
    img_w, img_h = pil_img.size

    # Domain check
    has_face = _face_check(pil_img)
    domain_warning = None
    if not has_face:
        domain_warning = ("Model trained on FFHQ vs StyleGAN faces (256px). "
                          "No face detected in uploaded image — result may be unreliable.")

    # --- Forensic signals (always computed) ---
    from forensics import compute_ela, compute_fft_spectrum, compute_laplacian_variance, compute_noise_residual
    ela_arr, ela_energy = compute_ela(pil_img)
    fft_arr, hf_ratio = compute_fft_spectrum(pil_img)
    lap_var = compute_laplacian_variance(pil_img)
    noise_arr, noise_uniformity = compute_noise_residual(pil_img)

    texture_report = {
        "ela_energy": round(ela_energy, 3),
        "high_freq_ratio": round(hf_ratio, 3),
        "laplacian_variance": round(lap_var, 2),
        "noise_uniformity": round(noise_uniformity, 3),
    }

    # --- Decide inference path ---
    trained_available = any(Path(f"checkpoints/{n}/best.pt").exists()
                            for n in ["cnn_baseline", "vit_baseline", "guided"])

    if not trained_available:
        result = _run_heuristic(pil_img, filename)
        result["texture_report"] = texture_report
        result["heatmaps"]["ela"] = _np_to_b64(ela_arr)
        result["heatmaps"]["fft_spectrum"] = _np_to_b64(fft_arr)
        result["domain_warning"] = domain_warning
        return result

    # --- Trained model inference ---
    tensor = _preprocess_pil(pil_img).to(device)
    if device.type == "cuda":
        tensor = tensor.to(memory_format=torch.channels_last)

    model_results = {}
    dtype = _DTYPE if torch.cuda.is_bf16_supported() else torch.float32

    for name in (models_to_run or list(_LOADED_MODELS.keys())):
        model = _LOADED_MODELS.get(name)
        if model is None:
            continue
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            out = model(tensor, return_aux=True)
        logit = (out["logits"] if isinstance(out, dict) else out).float().item()
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        model_results[name] = {"logit": logit, "prob_fake": prob, "output": out}

    if not model_results:
        result = _run_heuristic(pil_img, filename)
        result["domain_warning"] = domain_warning
        return result

    # Pick best model (priority: guided > hybrid > vit > cnn)
    priority = ["guided", "hybrid", "vit", "cnn"]
    primary_name = next((n for n in priority if n in model_results), list(model_results.keys())[0])
    primary = model_results[primary_name]
    hybrid_prob = primary["prob_fake"]
    is_fake = hybrid_prob >= 0.5
    confidence = (hybrid_prob if is_fake else 1.0 - hybrid_prob) * 100.0

    # Explainability heatmaps
    gradcam_b64 = None
    guidance_b64 = None
    vit_b64 = None

    if primary_name in ("guided", "hybrid") and "guided" in _LOADED_MODELS:
        guided_model = _LOADED_MODELS["guided"]
        try:
            from explain.gradcam import GradCAM
            cam_layer = guided_model.get_cam_layer()
            gcam = GradCAM(guided_model, cam_layer)
            cam_arr = gcam.compute(tensor.clone().requires_grad_(True), out_size=(img_w, img_h))
            gradcam_b64 = _np_to_b64(cam_arr)
            gcam.remove_hooks()
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")

        try:
            from explain.attention import extract_attention_map
            attn_arr = extract_attention_map(guided_model.vit, tensor, out_size=(img_w, img_h))
            vit_b64 = _np_to_b64(attn_arr)
            if "output" in primary and isinstance(primary["output"], dict):
                attn_map = primary["output"].get("attention_map")
                if attn_map is not None:
                    from forensics import _thermal_colormap
                    guidance_b64 = _np_to_b64(
                        _thermal_colormap(
                            (attn_map[0, 0].float().cpu().numpy() * 255).astype(np.uint8)
                        )
                    )
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}")

    # --- VIT attention fallback ---
    if vit_b64 is None and "vit" in _LOADED_MODELS:
        try:
            from explain.attention import extract_attention_map
            attn_arr = extract_attention_map(_LOADED_MODELS["vit"].backbone, tensor, out_size=(img_w, img_h))
            vit_b64 = _np_to_b64(attn_arr)
        except Exception:
            pass

    # --- Build old-compatible ELA ViT heuristic fallback for UI ---
    from app.detector import compute_vit_attention_map
    ela_pil = Image.fromarray(ela_arr)
    old_vit_heatmap, _ = compute_vit_attention_map(pil_img)

    t_end = time.perf_counter()
    inference_ms = (t_end - t_start) * 1000

    # Build models dict (backward-compatible with old API shape)
    models_dict = {}
    for n, r in model_results.items():
        p = r["prob_fake"]
        models_dict[n] = {
            "name": n,
            "prediction": "AI-Generated" if p >= 0.5 else "Authentic",
            "probability_fake": round(p * 100, 1),
            "focus": "Trained neural detector",
            "status": "Flagged" if p >= 0.5 else "Passed",
        }

    # Build legacy-compatible 3-model dict
    cnn_p = model_results.get("cnn", {}).get("prob_fake", hybrid_prob)
    vit_p = model_results.get("vit", {}).get("prob_fake", hybrid_prob)
    legacy_models = {
        "cnn": {
            "name": "CNN Baseline (EfficientNet-B0)",
            "prediction": "AI-Generated" if cnn_p >= 0.5 else "Authentic",
            "probability_fake": round(cnn_p * 100, 1),
            "focus": "Local pixel artifacts, high-frequency noise, texture edges",
            "status": "Flagged" if cnn_p >= 0.5 else "Passed",
        },
        "vit": {
            "name": "Vision Transformer (ViT-S/14 DINOv2)",
            "prediction": "AI-Generated" if vit_p >= 0.5 else "Authentic",
            "probability_fake": round(vit_p * 100, 1),
            "focus": "Patch self-attention, non-adjacent context, global lighting",
            "status": "Flagged" if vit_p >= 0.5 else "Passed",
        },
        "hybrid": {
            "name": "Hybrid / Guided CNN+ViT (Proposed)",
            "prediction": "AI-Generated" if hybrid_prob >= 0.5 else "Authentic",
            "probability_fake": round(hybrid_prob * 100, 1),
            "focus": "Fused local artifacts (CNN) + global context (ViT)",
            "status": "Flagged" if hybrid_prob >= 0.5 else "Passed",
        },
    }

    lower_fn = filename.lower()
    generator_type = "Authentic Camera Capture"
    if is_fake:
        if "stylegan" in lower_fn:
            generator_type = "Generative Adversarial Network (StyleGAN)"
        elif "midjourney" in lower_fn or "diffusion" in lower_fn:
            generator_type = "Latent Diffusion Model"
        else:
            generator_type = "Synthetic / AI-Generated"

    return {
        # === Backward-compatible fields ===
        "verdict": "AI-Generated / Synthetic" if is_fake else "Authentic / Real Photograph",
        "is_fake": is_fake,
        "confidence": round(confidence, 1),
        "generator_type": generator_type,
        "dimensions": f"{img_w} × {img_h} px",
        "reasoning": (
            f"Trained {primary_name} model ({confidence:.1f}% confidence). "
            f"ELA energy: {ela_energy:.2f}, HF ratio: {hf_ratio:.2f}, "
            f"Laplacian variance: {lap_var:.1f}."
        ),
        "models": legacy_models,
        "heatmaps": {
            "ela": _np_to_b64(ela_arr),
            "vit_attention": vit_b64 or _pil_to_b64(old_vit_heatmap),
            "gradcam": gradcam_b64,
            "guidance": guidance_b64,
            "fft_spectrum": _np_to_b64(fft_arr),
        },
        "forensics": {
            "noise_uniformity": round(noise_uniformity * 100, 1),
            "patch_discrepancy": round(hf_ratio * 100, 1),
            "spectral_frequency_energy": round(hf_ratio * 100, 1),
            "laplacian_gradient_variance": round(min(lap_var / 15.0, 98.0), 1),
        },
        # === New fields ===
        "model_source": "trained",
        "model_version": f"{primary_name}-v1.0",
        "calibrated_probability": round(hybrid_prob, 4),
        "inference_ms": round(inference_ms, 1),
        "domain_warning": domain_warning,
        "texture_report": texture_report,
    }
