"""
eval/run_eval.py
Full evaluation suite:
- Accuracy, Precision, Recall, F1, AUC-ROC, AP, EER, ECE
- Confusion matrices per model
- Robustness table: JPEG compression (q95/80/60), Gaussian blur, 0.5x down-up resize, Gaussian noise
- Temperature scaling calibration (threshold and temperature on val only)
- Latency benchmarking (median + p95 ms/image, warm GPU)
- OOD scoring via --ood-dir
- Saves results to reports/results.json
"""

import argparse
import json
import time
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robustness transforms using albumentations
# ---------------------------------------------------------------------------

def get_robustness_transform(corruption: str, crop_size: int = 224):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    base = [A.CenterCrop(height=crop_size, width=crop_size)]

    if corruption == "none":
        augments = []
    elif corruption == "jpeg95":
        augments = [A.ImageCompression(quality_range=(95, 95), p=1.0)]
    elif corruption == "jpeg80":
        augments = [A.ImageCompression(quality_range=(80, 80), p=1.0)]
    elif corruption == "jpeg60":
        augments = [A.ImageCompression(quality_range=(60, 60), p=1.0)]
    elif corruption == "blur":
        augments = [A.GaussianBlur(blur_limit=(5, 5), p=1.0)]
    elif corruption == "resize_half":
        augments = [A.Downscale(scale_range=(0.5, 0.5), p=1.0)]
    elif corruption == "noise":
        augments = [A.GaussNoise(p=1.0)]
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    return A.Compose(base + augments + [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def compute_eer(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def calibrate_temperature_and_threshold(model: nn.Module, val_loader: DataLoader,
                                         device: torch.device) -> tuple[float, float]:
    """Fits temperature and finds optimal classification threshold on validation set only."""
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    all_logits, all_labels = [], []
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                out = model(imgs)
            logits = out["logits"] if isinstance(out, dict) else out
            all_logits.append(logits.float())
            all_labels.append(lbls.float())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    def eval_step():
        optimizer.zero_grad()
        loss = criterion(scaler(all_logits), all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    T = float(scaler.temperature.item())

    # Find optimal decision threshold on validation set (best F1)
    probs_val = torch.sigmoid(all_logits / T).cpu().numpy()
    labels_val = all_labels.cpu().numpy()

    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(0.05, 0.95, 91):
        preds = (probs_val >= thresh).astype(int)
        f1 = f1_score(labels_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    logger.info(f"Calibrated on Validation: Temperature={T:.4f}, Optimal Threshold={best_thresh:.4f} (Val F1={best_f1:.4f})")
    return T, best_thresh


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(model: nn.Module, device: torch.device, n_warmup: int = 20,
                       n_runs: int = 200, batch_size: int = 1) -> Dict[str, float]:
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    with torch.no_grad():
        for _ in range(n_warmup):
            with torch.autocast(device_type=device.type, dtype=dtype):
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=dtype):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000 / batch_size)

    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


# ---------------------------------------------------------------------------
# Full eval on a split
# ---------------------------------------------------------------------------

def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device,
                   temperature: float = 1.0, threshold: float = 0.5) -> Dict[str, Any]:
    model.eval()
    all_logits, all_labels = [], []
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            lbls = batch["label"]
            with torch.autocast(device_type=device.type, dtype=dtype):
                out = model(imgs)
            logits = out["logits"] if isinstance(out, dict) else out
            all_logits.append(logits.float().cpu())
            all_labels.append(lbls)

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_np / temperature))
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels_np, preds).tolist()
    return {
        "accuracy": float(accuracy_score(labels_np, preds)),
        "precision": float(precision_score(labels_np, preds, zero_division=0)),
        "recall": float(recall_score(labels_np, preds, zero_division=0)),
        "f1": float(f1_score(labels_np, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(labels_np, probs)),
        "ap": float(average_precision_score(labels_np, probs)),
        "eer": compute_eer(probs, labels_np),
        "ece": compute_ece(probs, labels_np),
        "confusion_matrix": cm,
        "n_samples": int(len(labels_np)),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--model-name", required=True, help="Model name: cnn|vit|hybrid|guided")
    p.add_argument("--config", required=True, help="Path to YAML config used for training")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--ood-dir", default=None, help="Directory of OOD images to score")
    p.add_argument("--out", default="reports/results.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    # Load model
    from models.registry import build_model
    mcfg = dict(cfg["model"])
    mcfg.pop("name", None)
    model = build_model(args.model_name, **mcfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("ema_state", ckpt.get("model_state", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Build dataloaders
    from data.manifest import ManifestDataset, get_eval_transforms
    d = cfg["data"]
    crop = d.get("crop_size", 224)
    loader_kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)

    if args.split == "val":
        ds = ManifestDataset(d["val_manifest"], d["val_bin"], transforms=get_eval_transforms(crop))
    else:
        ds = ManifestDataset(d["test_manifest"], d["test_bin"], transforms=get_eval_transforms(crop))
    loader = DataLoader(ds, batch_size=64, shuffle=False, **loader_kw)

    val_ds = ManifestDataset(d["val_manifest"], d["val_bin"], transforms=get_eval_transforms(crop))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, **loader_kw)

    # Temperature and threshold calibration on val set
    temperature, threshold = calibrate_temperature_and_threshold(model, val_loader, device)

    # Main eval
    logger.info(f"Evaluating on {args.split} split with threshold={threshold:.4f}...")
    metrics = evaluate_split(model, loader, device, temperature=temperature, threshold=threshold)
    logger.info(f"Results: acc={metrics['accuracy']:.4f} auc={metrics['auc_roc']:.4f} "
                f"f1={metrics['f1']:.4f} ece={metrics['ece']:.4f}")

    # Robustness table
    corruptions = ["none", "jpeg95", "jpeg80", "jpeg60", "blur", "resize_half", "noise"]
    robustness = {}
    logger.info("Running robustness evaluation...")
    for corr in corruptions:
        c_ds = ManifestDataset(
            d["test_manifest"], d["test_bin"],
            transforms=get_robustness_transform(corr, crop_size=crop),
            subset=2000,
        )
        c_loader = DataLoader(c_ds, batch_size=64, shuffle=False, **loader_kw)
        c_metrics = evaluate_split(model, c_loader, device, temperature=temperature, threshold=threshold)
        robustness[corr] = {
            "accuracy": c_metrics["accuracy"],
            "auc_roc": c_metrics["auc_roc"],
        }
        logger.info(f"  {corr:15s}: acc={c_metrics['accuracy']:.4f} auc={c_metrics['auc_roc']:.4f}")

    # Latency
    logger.info("Benchmarking latency...")
    latency = benchmark_latency(model, device)
    logger.info(f"Latency: median={latency['median_ms']:.2f}ms p95={latency['p95_ms']:.2f}ms")

    # OOD scoring
    ood_scores = None
    if args.ood_dir:
        from PIL import Image
        import os, io
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        ood_tf = A.Compose([
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(height=crop, width=crop),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        ood_scores = []
        ood_dir = Path(args.ood_dir)
        for img_path in ood_dir.glob("*.*"):
            try:
                with Image.open(img_path) as im:
                    arr = np.array(im.convert("RGB"))
                t = ood_tf(image=arr)["image"].unsqueeze(0).to(device)
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                    out = model(t)
                logit = out["logits"].item() if isinstance(out, dict) else out.item()
                prob = 1.0 / (1.0 + np.exp(-logit / temperature))
                ood_scores.append({"file": img_path.name, "prob_fake": float(prob)})
            except Exception as e:
                logger.warning(f"Skipped OOD file {img_path.name}: {e}")

        logger.info(f"OOD scored {len(ood_scores)} images. "
                    f"Mean p(fake)={np.mean([s['prob_fake'] for s in ood_scores]):.3f}")

    # Compile full results
    results = {
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "dataset": "FFHQ vs StyleGAN faces (256px)",
        "split": args.split,
        "temperature": temperature,
        "metrics": metrics,
        "robustness": robustness,
        "latency": latency,
        "ood_scores": ood_scores,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    all_results = []
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                all_results = [r for r in existing if r.get("model_name") != args.model_name]
            elif isinstance(existing, dict):
                if existing.get("model_name") != args.model_name:
                    all_results = [existing]
        except Exception:
            all_results = []

    all_results.append(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results for '{args.model_name}' saved to {args.out} (total models: {len(all_results)})")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
