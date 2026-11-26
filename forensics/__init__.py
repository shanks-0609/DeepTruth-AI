"""
forensics/__init__.py
Pure-function forensic signal extractors (ELA, FFT, Laplacian, noise residual).
All functions are stateless and have no side effects.
"""

import io
import numpy as np
from PIL import Image, ImageChops


def compute_ela(pil_img: Image.Image, quality: int = 90, scale: int = 18) -> tuple[np.ndarray, float]:
    """
    Error Level Analysis.
    Returns (heatmap_array [H,W,3] uint8, ela_energy float).
    """
    img_rgb = pil_img.convert("RGB")
    buf = io.BytesIO()
    img_rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)
    diff = ImageChops.difference(img_rgb, resaved)
    diff_arr = np.array(diff, dtype=np.float32)
    gray_diff = np.mean(diff_arr, axis=2)
    scaled = np.clip(gray_diff * scale, 0, 255).astype(np.uint8)
    ela_energy = float(np.mean(gray_diff))
    heatmap = _thermal_colormap(scaled)
    return heatmap, ela_energy


def compute_fft_spectrum(pil_img: Image.Image) -> tuple[np.ndarray, float]:
    """
    2D FFT log-magnitude spectrum and high-frequency radial ratio.
    Returns (spectrum_array [H,W,3] uint8, hf_ratio float).
    """
    gray = np.array(pil_img.convert("L").resize((256, 256)), dtype=np.float32)
    fft2 = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft2)
    mag = np.log(np.abs(fft_shift) + 1.0)
    mag_norm = ((mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255).astype(np.uint8)
    spectrum_heatmap = _thermal_colormap(mag_norm)

    cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
    y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    hf_mask = r > (gray.shape[0] * 0.35)
    hf_ratio = float(np.mean(mag[hf_mask]) / (np.mean(mag) + 1e-6))
    return spectrum_heatmap, hf_ratio


def compute_laplacian_variance(pil_img: Image.Image) -> float:
    """
    Laplacian gradient variance — quantifies micro-edge sharpness.
    """
    from scipy.signal import convolve2d
    gray = np.array(pil_img.convert("L").resize((256, 256)), dtype=np.float32)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = convolve2d(gray, kernel, mode="valid")
    return float(np.var(lap))


def compute_noise_residual(pil_img: Image.Image) -> tuple[np.ndarray, float]:
    """
    Camera noise residual via high-pass filter. Returns (residual_heatmap, uniformity_score).
    """
    from scipy.ndimage import gaussian_filter
    rgb = np.array(pil_img.convert("RGB"), dtype=np.float32)
    smoothed = np.stack([gaussian_filter(rgb[:, :, c], sigma=2.0) for c in range(3)], axis=2)
    residual = np.abs(rgb - smoothed)
    gray_residual = np.mean(residual, axis=2)
    uniformity = float(1.0 - np.std(gray_residual) / (np.mean(gray_residual) + 1e-6))
    gray_norm = np.clip(gray_residual * 10, 0, 255).astype(np.uint8)
    heatmap = _thermal_colormap(gray_norm)
    return heatmap, uniformity


def _thermal_colormap(gray: np.ndarray) -> np.ndarray:
    """Maps 2D uint8 array to thermal RGB heatmap (Blue->Cyan->Yellow->Red)."""
    n = gray.astype(np.float32) / 255.0
    r = np.clip(1.5 * n - 0.2, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(n - 0.5) * 2.0, 0.0, 1.0)
    b = np.clip(1.0 - 2.0 * n, 0.0, 1.0)
    r = np.where(n > 0.8, 1.0, r)
    g = np.where(n > 0.8, 0.9, g)
    b = np.where(n > 0.8, 0.9, b)
    return np.stack([r * 255, g * 255, b * 255], axis=-1).astype(np.uint8)
