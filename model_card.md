# Model Card — DeepTruth AI v2.0

## Overview

| Field | Value |
|---|---|
| **Task** | Binary classification: Real (FFHQ camera) vs Fake (StyleGAN) faces |
| **Dataset** | [TheKernel01/140k-Real-and-Fake-Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces) |
| **Input** | 224×224 center-crop of 256×256 JPEG face images |
| **Output** | P(fake) ∈ [0, 1], binary verdict, attention heatmaps |
| **Domain** | Face images only. Non-face inputs receive a `domain_warning`. |
| **Hardware trained on** | NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, sm_120, 8 GB VRAM) |
| **Framework** | PyTorch 2.14.1+cu130, bfloat16, timm, Albumentations |

---

## Dataset

| Split | Images | Real (FFHQ) | Fake (StyleGAN) |
|---|---|---|---|
| Train | 100,000 | 50,000 | 50,000 |
| Validation | 20,000 | 10,000 | 10,000 |
| Test | 20,000 | 10,000 | 10,000 |

**Data leakage audit findings** (`reports/data_audit.md`):
- Metadata-only classifier (file size, color mean, JPEG quant tables): **55.0%** accuracy — no meaningful shortcut.
- Frequency-only baseline (FFT radial profile → logistic regression): **61.1%** — mild spectral signature from StyleGAN upsampling. Augmentations (JPEG re-compression q=50–95, blur, noise) applied to mitigate.
- Perceptual hash cross-split duplicates: **0 found**.

> [!WARNING]
> This dataset consists exclusively of StyleGAN-generated fakes vs FFHQ real faces. Models trained here **should not** be expected to generalise to other generators (Midjourney, Stable Diffusion, etc.) without Phase B retraining.

---

## Architectures Trained

### 1. CNN Baseline (`models/cnn_baseline.py`)
- **Backbone:** EfficientNet-B0 (`timm/efficientnet_b0.ra_in1k`)
- **Head:** Global average pool → Dropout(0.3) → Linear(1280 → 1)
- **Checkpoint:** `checkpoints/cnn_baseline/best.pt` (70 MB)
- **Training:** 15 epochs, 20k-image subset, AdamW, cosine LR, bf16

### 2. ViT Baseline (`models/vit_baseline.py`)
- **Backbone:** DINOv2 ViT-S/14 (`timm/vit_small_patch14_dinov2.lvd142m`)
- **Input resolution:** 224×224 (`dynamic_img_size=True`, 16×16 patches)
- **Head:** CLS token → Dropout(0.3) → Linear(384 → 1)
- **Checkpoint:** `checkpoints/vit_baseline/best.pt` (88 MB)
- **Training:** 6 epochs, 20k-image subset, bf16

### 3. ViT-Guided CNN — Proposed (`models/guided_net.py`)
- **CNN branch:** EfficientNet-B0 (texture / local artifacts)
- **ViT branch:** DINOv2 ViT-S/14 (global / semantic guidance)
- **Frequency stream:** Fixed SRM high-pass filters feeding a small CNN
- **Guidance mechanism:**
  - ViT attention rollout → spatial importance map A
  - Spatial gating: `F' = F * (1 + σ(conv(A)))` on CNN stage features
  - FiLM modulation: γ, β from ViT CLS token applied to CNN channels
  - Cross-attention: CNN feature tokens attend to ViT patch tokens
  - Fusion head: `[pool(CNN) ∥ CLS ∥ freq]` → MLP → 1 logit
- **Checkpoint:** `checkpoints/guided/best.pt` (323.9 MB)
- **Training:** Staged (freeze backbones → unfreeze CNN + LoRA ViT → joint low-LR), bf16, EMA

---

## Empirical Test Results (20,000 images, genuine measurement)

> [!IMPORTANT]
> All numbers below are measured on the **held-out test split** (never used during training or threshold selection). Temperature and threshold calibrated on validation only.

### Primary Metrics

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Latency (median) |
|---|---|---|---|---|---|---|
| CNN Baseline | 59.6% | 55.9% | 91.0% | 69.3% | 0.665 | 13.7 ms |
| ViT Baseline | 76.5% | 72.5% | 85.5% | 78.5% | 0.857 | 9.4 ms |
| **ViT-Guided CNN** | **88.5%** | **88.0%** | **89.2%** | **88.5%** | **0.956** | 25.1 ms |

### Confusion Matrices (Test Split, 20k images)

**CNN Baseline:**
```
              Pred Real  Pred Fake
Actual Real    2,820      7,180
Actual Fake      897      9,103
```

**ViT Baseline:**
```
              Pred Real  Pred Fake
Actual Real    6,760      3,240
Actual Fake    1,451      8,549
```

**ViT-Guided CNN (Proposed):**
```
              Pred Real  Pred Fake
Actual Real    8,779      1,221
Actual Fake    1,085      8,915
```

### Robustness Table (2,000 image subset per corruption)

| Corruption | CNN Acc | ViT Acc | Guided Acc |
|---|---|---|---|
| None (clean) | 60.7% | 77.4% | **88.8%** |
| JPEG q=95 | 60.3% | 77.3% | **88.5%** |
| JPEG q=80 | 59.7% | 74.5% | **86.3%** |
| JPEG q=60 | 59.8% | 72.3% | **83.9%** |
| Gaussian Blur | 59.9% | 75.9% | **85.8%** |
| Downscale ×0.5 | 61.1% | 75.0% | **82.9%** |
| Gaussian Noise | 60.2% | 69.5% | 56.3% |

> [!NOTE]
> The Guided CNN degrades significantly under Gaussian noise (56.3%). This is expected: the SRM and high-frequency artifact signals are most susceptible to additive noise. The ViT branch partially compensates but cannot fully recover. Denoising augmentation during Phase B training is recommended.

### Calibration

| Model | Temperature | Optimal Threshold | Val F1 |
|---|---|---|---|
| CNN | 1.158 | 0.590 | 0.689 |
| ViT | 0.973 | 0.580 | 0.788 |
| Guided | 0.738 | 0.690 | **0.885** |

---

## OOD (Out-of-Distribution) Scores

Images in `samples/` scored by each model — these are out-of-domain (non-FFHQ faces, Midjourney output). Treated as an illustrative check only, **not** a cross-generator generalisation claim.

| File | CNN P(fake) | ViT P(fake) | Guided P(fake) |
|---|---|---|---|
| stylegan_portrait.jpg | 0.679 | 0.628 | 0.685 |
| midjourney_astronaut.jpg | 0.653 | 0.405 | 0.511 |
| real_watchmaker.jpg | 0.424 | 0.361 | 0.536 |
| real_lake_mountain.jpg | 0.455 | 0.537 | 0.326 |

> [!CAUTION]
> OOD scores are unreliable. The model was trained exclusively on StyleGAN vs FFHQ. For non-face or non-StyleGAN inputs, scores are essentially random. The API returns a `domain_warning` for non-face inputs.

---

## Training Configuration

See `configs/guided.yaml` for the full YAML. Key hyperparameters:

| Parameter | Value |
|---|---|
| Input size | 224×224 (native crop from 256px, no resampling) |
| Effective batch size | 64 (micro-batch 16 × 4 grad accumulation) |
| Precision | bfloat16 autocast |
| Optimizer | AdamW (β₁=0.9, β₂=0.999, weight decay=0.01) |
| LR (head) | 1e-4 |
| LR (backbones) | 1e-5 |
| Schedule | Cosine with 500-step warmup |
| Gradient clipping | max_norm=1.0 |
| Label smoothing | 0.05 |
| EMA decay | 0.9998 |
| Loss | BCE with logits |
| Peak VRAM | < 2.0 GB (well within 7.5 GB gate) |
| Augmentations | JPEG re-compress (q=50–95), Gaussian blur, down-up resize, Gaussian noise, horizontal flip, mild color jitter |

---

## Limitations

1. **Single generator**: Trained only on FFHQ (real) vs StyleGAN (fake). Do not use for Stable Diffusion, Midjourney, or other diffusion-model outputs without Phase B retraining on those distributions.
2. **Face-only domain**: The model is calibrated for 256px face images. Performance on non-face images is undefined. The API issues a `domain_warning` when no face is detected.
3. **Noise vulnerability**: Gaussian noise drastically drops Guided CNN accuracy (56.3%). Real-world noisy photos should be pre-processed.
4. **Frequency leakage**: 61.1% FFT-only baseline confirms spectral shortcuts exist in this dataset. While JPEG/noise augmentation mitigates this, the model may still exploit spectral cues rather than purely semantic ones.
5. **Subset training**: Models were trained on 20k-image subsets (not the full 100k). Full-dataset training is expected to improve generalisation.

---

## Intended Use

- **Research & academic demonstration** of ViT-guided attention mechanisms for forensic artifact detection.
- **Semester 5 IMD project**, Bennett University.
- **Not intended** for high-stakes real-world deployment without further validation.

---

## Citation / Dataset

```
@dataset{TheKernel01_140k_faces,
  author = {TheKernel01},
  title  = {140k Real and Fake Faces},
  year   = {2023},
  url    = {https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces},
}
```
