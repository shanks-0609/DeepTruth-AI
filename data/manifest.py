"""
data/manifest.py
Defines the ManifestDataset interface, data loading structures, and transform pipelines.
Supports memory-mapped encoded JPEG shards, native 224x224 crops without resampling,
and Albumentations degradation augmentations for artifact detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import io
import json
import mmap
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class ManifestItem:
    index: int
    split: str
    label: int  # 0: Real, 1: Fake
    generator: int  # 0: Real, 1: StyleGAN, etc.
    path: str
    byte_offset: int
    byte_length: int


def get_train_transforms(crop_size: int = 224) -> A.Compose:
    """
    Native-resolution crops with degradation augmentations (JPEG recompression, blur, noise).
    NO geometric scaling/resampling that destroys high-frequency artifact signatures.
    """
    return A.Compose([
        A.RandomCrop(height=crop_size, width=crop_size),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ImageCompression(quality_range=(50, 95), p=0.7),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.GaussNoise(p=0.5),
        ], p=0.6),
        A.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.04, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_eval_transforms(crop_size: int = 224) -> A.Compose:
    """
    Deterministic center-crop with standard ImageNet normalization. No resampling.
    """
    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class ManifestDataset(Dataset):
    """
    Generic dataset reading from manifest JSONL and memory-mapped binary shards.
    Keeps encoded image bytes in mmap; decodes on-the-fly in worker processes.
    """

    def __init__(
        self,
        manifest_path: str,
        bin_path: str,
        transforms: Optional[Callable] = None,
        subset: Optional[int] = None,
        seed: int = 42,
    ):
        self.manifest_path = Path(manifest_path)
        self.bin_path = Path(bin_path)
        self.transforms = transforms
        self.items: List[ManifestItem] = []

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary shard not found: {self.bin_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.items.append(
                    ManifestItem(
                        index=d["index"],
                        split=d["split"],
                        label=int(d["label"]),
                        generator=int(d.get("generator", d["label"])),
                        path=d.get("path", f"{d['index']}.jpg"),
                        byte_offset=d["byte_offset"],
                        byte_length=d["byte_length"],
                    )
                )

        if subset is not None and subset > 0 and subset < len(self.items):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(self.items), size=subset, replace=False)
            self.items = [self.items[i] for i in sorted(indices)]

        self._mmap_file = None
        self._mmap_obj = None

    def _ensure_mmap(self):
        if self._mmap_obj is None:
            self._mmap_file = open(self.bin_path, "rb")
            self._mmap_obj = mmap.mmap(
                self._mmap_file.fileno(), 0, access=mmap.ACCESS_READ
            )

    def __len__(self) -> int:
        return len(self.items)

    def get_raw_bytes(self, idx: int) -> bytes:
        self._ensure_mmap()
        item = self.items[idx]
        self._mmap_obj.seek(item.byte_offset)
        return self._mmap_obj.read(item.byte_length)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        raw_bytes = self.get_raw_bytes(idx)

        # Decode JPEG bytes with PIL
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            img = pil_img.convert("RGB")
            img_np = np.array(img, dtype=np.uint8)

        if self.transforms is not None:
            augmented = self.transforms(image=img_np)
            tensor_img = augmented["image"]
        else:
            tensor_img = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0

        return {
            "image": tensor_img,
            "label": torch.tensor(item.label, dtype=torch.float32),
            "generator": torch.tensor(item.generator, dtype=torch.long),
            "index": item.index,
        }

    def __del__(self):
        if self._mmap_obj is not None:
            try:
                self._mmap_obj.close()
            except Exception:
                pass
        if self._mmap_file is not None:
            try:
                self._mmap_file.close()
            except Exception:
                pass
