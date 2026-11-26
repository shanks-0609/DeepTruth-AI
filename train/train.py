"""
train/train.py
Entry-point for training any model in the registry.
Usage:
    python -m train.train --config configs/guided.yaml [--subset 20000] [--resume checkpoints/guided/best.pt]

Properly guarded by  if __name__ == '__main__'  for Windows multiprocessing spawn safety.
"""

import argparse
import sys
import logging
from pathlib import Path

import yaml
import torch

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="DeepTruth AI — Model Training")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--subset", type=int, default=None,
                   help="Train on N images only (fast experiment mode)")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.subset is not None:
        cfg.setdefault("data", {})["subset"] = args.subset
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Import here so multiprocessing spawn won't re-execute this block
    from train.trainer import Trainer

    subset = cfg.get("data", {}).get("subset", None)
    trainer = Trainer(cfg)

    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {cfg['model']['name']} | Subset: {subset} | Device: "
                f"{'cuda:0' if torch.cuda.is_available() else 'cpu'}")

    trainer.run(subset=subset)


if __name__ == "__main__":
    # REQUIRED for Windows DataLoader with num_workers > 0 (spawn-based multiprocessing)
    import multiprocessing
    multiprocessing.freeze_support()
    main()
