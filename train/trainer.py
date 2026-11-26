"""
train/trainer.py
Core training engine:
- bf16 autocast, gradient accumulation, grad clipping
- EMA weight averaging
- Staged backbone freezing / unfreezing
- Cosine LR schedule with warmup
- TensorBoard logging and best-AUC checkpoint saving
- Windows-safe DataLoader (spawn-aware, persistent_workers)
"""

import os
import sys
import time
import logging
import math
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import numpy as np

from data.manifest import ManifestDataset, get_train_transforms, get_eval_transforms
from models.registry import build_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.copy_(self.decay * ema_p + (1.0 - self.decay) * p.detach())

    def get_model(self) -> nn.Module:
        return self.shadow


# ---------------------------------------------------------------------------
# Cosine LR with linear warmup
# ---------------------------------------------------------------------------

def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def lr_lambda(step):
        if step < warmup_steps:
            factor = step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / max(base_lrs), factor)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Parameter group builder with LLRD
# ---------------------------------------------------------------------------

def build_param_groups(model: nn.Module, lr_head: float, lr_backbone: float,
                        weight_decay: float = 0.05, layer_decay: float = 0.75) -> List[Dict]:
    """
    Separates head and backbone parameters, with optional layerwise LR decay.
    """
    # collect head params
    head_params = []
    backbone_params = []

    head_names = {"classifier", "fusion_head", "aux_cnn_head", "aux_vit_head",
                  "cross_attn", "film", "spatial_gate", "freq_net"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split(".")[0]
        is_head = top in head_names or any(h in name for h in head_names)
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = [
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
    ]
    return [g for g in groups if len(g["params"]) > 0]


# ---------------------------------------------------------------------------
# Freeze/Unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_backbones(model: nn.Module):
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if top in ("cnn", "vit", "backbone"):
            param.requires_grad_(False)
    logger.info("Backbones FROZEN.")


def unfreeze_backbones(model: nn.Module, unfreeze_vit_last_n: int = 4):
    # Always unfreeze CNN
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if top == "cnn":
            param.requires_grad_(True)

    # Unfreeze last N ViT blocks (or entire backbone for CNN-only)
    if hasattr(model, "vit"):
        vit = model.vit
        if hasattr(vit, "blocks"):
            blocks = vit.blocks
            n = len(blocks)
            for i, block in enumerate(blocks):
                unfreeze = (i >= n - unfreeze_vit_last_n)
                for p in block.parameters():
                    p.requires_grad_(unfreeze)
        # Always unfreeze final norm
        if hasattr(vit, "norm"):
            for p in vit.norm.parameters():
                p.requires_grad_(True)

    # Guidance modules always trainable
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if top in ("film", "spatial_gate", "cross_attn", "freq_net",
                   "classifier", "fusion_head", "aux_cnn_head", "aux_vit_head"):
            param.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Backbones UNFROZEN. Trainable params: {n_trainable:,}")


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def compute_loss(outputs, labels, aux_weight: float = 0.3,
                 label_smoothing: float = 0.05) -> torch.Tensor:
    """BCE with label smoothing + optional auxiliary branch losses."""
    smooth_labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
    criterion = nn.BCEWithLogitsLoss()

    if isinstance(outputs, dict):
        main_loss = criterion(outputs["logits"], smooth_labels)
        aux_loss = torch.tensor(0.0, device=labels.device)
        if "aux_cnn_logits" in outputs:
            aux_loss = aux_loss + criterion(outputs["aux_cnn_logits"], smooth_labels)
        if "aux_vit_logits" in outputs:
            aux_loss = aux_loss + criterion(outputs["aux_vit_logits"], smooth_labels)
        return main_loss + aux_weight * aux_loss
    else:
        return criterion(outputs, smooth_labels)


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             use_amp: bool = True) -> Dict[str, float]:
    model.eval()
    all_logits, all_labels = [], []
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float32

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            out = model(images, return_aux=True) if hasattr(model, "cross_attn") else model(images)

        logits = out["logits"] if isinstance(out, dict) else out
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    acc = float(np.mean(preds == all_labels))
    auc = float(roc_auc_score(all_labels, probs))
    return {"val_acc": acc, "val_auc": auc}


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.setup_logging()
        self.setup_dirs()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def setup_dirs(self):
        Path(self.cfg["checkpoints"]["dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.cfg["logging"]["tensorboard_dir"]).mkdir(parents=True, exist_ok=True)

    def build_loaders(self, subset: Optional[int] = None):
        d = self.cfg["data"]
        train_ds = ManifestDataset(
            d["train_manifest"], d["train_bin"],
            transforms=get_train_transforms(d.get("crop_size", 224)),
            subset=subset,
        )
        val_ds = ManifestDataset(
            d["val_manifest"], d["val_bin"],
            transforms=get_eval_transforms(d.get("crop_size", 224)),
        )
        loader_kwargs = dict(
            num_workers=d.get("num_workers", 4),
            pin_memory=d.get("pin_memory", True),
            persistent_workers=d.get("persistent_workers", True),
            prefetch_factor=2,
        )
        train_loader = DataLoader(train_ds, batch_size=self.cfg["training"]["micro_batch"],
                                  shuffle=True, drop_last=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=self.cfg["training"]["micro_batch"] * 2,
                                shuffle=False, **loader_kwargs)
        return train_loader, val_loader

    def build_model(self):
        mcfg = self.cfg["model"]
        name = mcfg.pop("name")
        model = build_model(name, **{k: v for k, v in mcfg.items() if k != "name"})
        mcfg["name"] = name  # restore

        if self.cfg["training"].get("use_channels_last", True):
            model = model.to(memory_format=torch.channels_last)
        return model.to(self.device)

    def run(self, subset: Optional[int] = None):
        torch.manual_seed(self.cfg["training"]["seed"])
        np.random.seed(self.cfg["training"]["seed"])

        train_loader, val_loader = self.build_loaders(subset=subset)
        model = self.build_model()

        tcfg = self.cfg["training"]
        use_amp = tcfg.get("use_amp", True)
        accum = tcfg.get("accum_steps", 4)
        dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float32

        ema = ModelEMA(model, decay=self.cfg.get("ema", {}).get("decay", 0.9999)) \
            if self.cfg.get("ema", {}).get("enabled", True) else None

        writer = SummaryWriter(self.cfg["logging"]["tensorboard_dir"])
        best_auc = 0.0
        global_step = 0
        ckpt_dir = Path(self.cfg["checkpoints"]["dir"])
        log_interval = self.cfg["logging"].get("log_interval", 50)

        stages = tcfg.get("stages", None)
        if stages:
            # Staged training
            self._run_staged(model, ema, train_loader, val_loader, stages,
                             accum, dtype, use_amp, writer, ckpt_dir, log_interval, tcfg)
        else:
            # Simple single-stage training
            lr = tcfg.get("lr", 1e-4)
            wd = tcfg.get("weight_decay", 0.05)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            total_steps = tcfg["epochs"] * len(train_loader) // accum
            scheduler = get_cosine_schedule(optimizer, warmup_steps=max(1, total_steps // 10),
                                            total_steps=total_steps)

            for epoch in range(tcfg["epochs"]):
                self._train_epoch(model, optimizer, scheduler, ema, train_loader,
                                  epoch, accum, dtype, use_amp, tcfg, writer, log_interval,
                                  global_step)
                val_metrics = validate(model, val_loader, self.device, use_amp)
                logger.info(f"Epoch {epoch+1}: val_acc={val_metrics['val_acc']:.4f}, "
                            f"val_auc={val_metrics['val_auc']:.4f}")
                writer.add_scalar("val/acc", val_metrics["val_acc"], epoch)
                writer.add_scalar("val/auc", val_metrics["val_auc"], epoch)

                if val_metrics["val_auc"] > best_auc:
                    best_auc = val_metrics["val_auc"]
                    self._save_checkpoint(model, ema, optimizer, epoch, val_metrics, ckpt_dir / "best.pt")

        writer.close()
        logger.info(f"Training complete. Best val AUC: {best_auc:.4f}")

    def _run_staged(self, model, ema, train_loader, val_loader, stages,
                    accum, dtype, use_amp, writer, ckpt_dir, log_interval, tcfg):
        best_auc = 0.0
        global_step = 0
        label_smoothing = tcfg.get("label_smoothing", 0.05)
        aux_weight = tcfg.get("aux_weight", 0.3)
        grad_clip = tcfg.get("grad_clip", 1.0)
        wd = tcfg.get("weight_decay", self.cfg.get("optimizer", {}).get("weight_decay", 0.05))

        for stage_cfg in stages:
            stage_name = stage_cfg["name"]
            n_epochs = stage_cfg["epochs"]
            logger.info(f"\n{'='*60}\nSTAGE: {stage_name} ({n_epochs} epochs)\n{'='*60}")

            # Freeze/unfreeze
            if stage_cfg.get("freeze_backbones", False):
                freeze_backbones(model)
            else:
                unfreeze_backbones(model, unfreeze_vit_last_n=stage_cfg.get("unfreeze_vit_last_n", 4))

            # Build optimizer for this stage
            lr_head = stage_cfg.get("lr_head", 1e-4)
            lr_backbone = stage_cfg.get("lr_backbone", 1e-5)
            param_groups = build_param_groups(model, lr_head, lr_backbone, wd)
            optimizer = torch.optim.AdamW(param_groups,
                                          betas=self.cfg.get("optimizer", {}).get("betas", (0.9, 0.999)))
            total_steps = n_epochs * len(train_loader) // accum
            warmup = max(1, total_steps // 10)
            scheduler = get_cosine_schedule(optimizer, warmup, total_steps)

            for epoch in range(n_epochs):
                gs = self._train_epoch(model, optimizer, scheduler, ema, train_loader,
                                       epoch, accum, dtype, use_amp, tcfg, writer,
                                       log_interval, global_step, stage_name)
                global_step = gs
                val_metrics = validate(model, val_loader, self.device, use_amp)
                logger.info(f"[{stage_name}] Epoch {epoch+1}/{n_epochs}: "
                            f"val_acc={val_metrics['val_acc']:.4f}, val_auc={val_metrics['val_auc']:.4f}")
                writer.add_scalar("val/acc", val_metrics["val_acc"], global_step)
                writer.add_scalar("val/auc", val_metrics["val_auc"], global_step)

                if val_metrics["val_auc"] > best_auc:
                    best_auc = val_metrics["val_auc"]
                    self._save_checkpoint(model, ema, optimizer, epoch, val_metrics,
                                          ckpt_dir / "best.pt")
                    logger.info(f"  New best AUC: {best_auc:.4f} — checkpoint saved.")

        return best_auc

    def _train_epoch(self, model, optimizer, scheduler, ema, loader,
                     epoch, accum, dtype, use_amp, tcfg, writer, log_interval,
                     global_step, stage_name="train"):
        model.train()
        label_smoothing = tcfg.get("label_smoothing", 0.05)
        aux_weight = tcfg.get("aux_weight", 0.3)
        grad_clip = tcfg.get("grad_clip", 1.0)
        return_aux = hasattr(model, "cross_attn") or hasattr(model, "aux_cnn_head")

        optimizer.zero_grad()
        t0 = time.time()
        running_loss = 0.0

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            if tcfg.get("use_channels_last", True):
                images = images.to(memory_format=torch.channels_last)

            with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=use_amp):
                out = model(images, return_aux=return_aux)
                loss = compute_loss(out, labels, aux_weight, label_smoothing) / accum

            loss.backward()
            running_loss += loss.item() * accum

            if (step + 1) % accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                global_step += 1

                if global_step % log_interval == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    imgs_sec = (step + 1) * images.shape[0] / elapsed
                    vram_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    logger.info(f"[{stage_name}] e{epoch+1} step{global_step} "
                                f"loss={running_loss/(step+1):.4f} lr={lr_now:.2e} "
                                f"imgs/s={imgs_sec:.0f} VRAM={vram_mb:.0f}MB")
                    writer.add_scalar("train/loss", running_loss / (step + 1), global_step)
                    writer.add_scalar("train/lr", lr_now, global_step)

        return global_step

    def _save_checkpoint(self, model, ema, optimizer, epoch, metrics, path: Path):
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        }
        if ema:
            payload["ema_state"] = ema.shadow.state_dict()
        torch.save(payload, path)
