"""
app/evaluation.py
Reads empirical benchmark metrics from reports/results.json.
Removes all fabricated/hardcoded numbers and reflects genuine experimental measurements.
"""

from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

RESULTS_FILE = Path(__file__).resolve().parent.parent / "reports" / "results.json"


def get_benchmark_data() -> dict:
    """
    Returns empirical benchmark evaluation metrics from reports/results.json.
    If evaluation has not yet been executed, returns an honest status payload.
    """
    if not RESULTS_FILE.exists():
        return {
            "status": "pending",
            "message": "Evaluation suite has not been executed yet. Run eval/run_eval.py to generate genuine metrics.",
            "dataset_summary": {
                "name": "Hugging Face TheKernel01/140k-Real-and-Fake-Faces",
                "description": "FFHQ vs StyleGAN faces (256px)",
                "total_images": 140000,
                "train_set": 100000,
                "val_set": 20000,
                "test_set": 20000,
                "classes": ["Real (FFHQ Camera)", "Fake (StyleGAN)"],
            },
            "model_comparison": [],
            "confusion_matrices": {},
            "generator_breakdown": [],
        }

    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        models_data = data if isinstance(data, list) else [data]

        model_comparison = []
        confusion_matrices = {}

        for m in models_data:
            m_name = m.get("model_name", "model")
            metrics = m.get("metrics", {})
            lat = m.get("latency", {})
            cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

            tn = int(cm[0][0]) if len(cm) > 0 and len(cm[0]) > 0 else 0
            fp = int(cm[0][1]) if len(cm) > 0 and len(cm[0]) > 1 else 0
            fn = int(cm[1][0]) if len(cm) > 1 and len(cm[1]) > 0 else 0
            tp = int(cm[1][1]) if len(cm) > 1 and len(cm[1]) > 1 else 0

            display_name = {
                "cnn": "CNN Baseline (EfficientNet-B0)",
                "vit": "Vision Transformer (ViT-S/14 DINOv2)",
                "hybrid": "Concat-Hybrid (CNN + ViT)",
                "guided": "ViT-Guided CNN (Proposed)",
            }.get(m_name, m_name.upper())

            backbone_desc = {
                "cnn": "EfficientNet-B0 (penultimate pooling)",
                "vit": "ViT-S/14 DINOv2 (patch tokens + CLS)",
                "hybrid": "Late concatenation [z_CNN || z_ViT]",
                "guided": "Spatial Attention Gating + FiLM + Cross-Attention",
            }.get(m_name, m.get("checkpoint", "Custom"))

            model_comparison.append({
                "id": m_name,
                "name": display_name,
                "backbone": backbone_desc,
                "accuracy": round(metrics.get("accuracy", 0.0) * 100, 2),
                "precision": round(metrics.get("precision", 0.0) * 100, 2),
                "recall": round(metrics.get("recall", 0.0) * 100, 2),
                "f1_score": round(metrics.get("f1", 0.0) * 100, 2),
                "auc_roc": round(metrics.get("auc_roc", 0.0), 4),
                "latency_ms": round(lat.get("median_ms", 0.0), 1),
            })

            confusion_matrices[m_name] = {
                "model_name": display_name,
                "test_total": tn + fp + fn + tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            }

        # Build a dict keyed by model_name for easy cross-model robustness lookup
        by_model = {m.get("model_name", ""): m for m in models_data}

        def _rob(model_id: str, corruption: str) -> float:
            m = by_model.get(model_id, {})
            return round(m.get("robustness", {}).get(corruption, {}).get("accuracy", 0.0) * 100, 1)

        def _acc(model_id: str) -> float:
            m = by_model.get(model_id, {})
            return round(m.get("metrics", {}).get("accuracy", 0.0) * 100, 1)

        generator_breakdown = [
            {
                "generator": "Clean Test Split (No Corruption)",
                "cnn_acc": _rob("cnn", "none") or _acc("cnn"),
                "vit_acc": _rob("vit", "none") or _acc("vit"),
                "hybrid_acc": _rob("hybrid", "none") or _acc("guided"),
                "guided_acc": _rob("guided", "none") or _acc("guided"),
                "notes": "Evaluated on genuine 20k test split without degradation",
            },
            {
                "generator": "JPEG Quality 80 (Lossy Compression)",
                "cnn_acc": _rob("cnn", "jpeg80"),
                "vit_acc": _rob("vit", "jpeg80"),
                "hybrid_acc": _rob("hybrid", "jpeg80"),
                "guided_acc": _rob("guided", "jpeg80"),
                "notes": "Tests resilience to social media JPEG recompression",
            },
            {
                "generator": "Gaussian Blur (Smoothing)",
                "cnn_acc": _rob("cnn", "blur"),
                "vit_acc": _rob("vit", "blur"),
                "hybrid_acc": _rob("hybrid", "blur"),
                "guided_acc": _rob("guided", "blur"),
                "notes": "Tests resilience against post-processing smoothing filters",
            },
            {
                "generator": "Gaussian Noise (Additive)",
                "cnn_acc": _rob("cnn", "noise"),
                "vit_acc": _rob("vit", "noise"),
                "hybrid_acc": _rob("hybrid", "noise"),
                "guided_acc": _rob("guided", "noise"),
                "notes": "Tests stability against simulated sensor noise",
            },
        ]

        return {
            "status": "ready",
            "dataset_summary": {
                "name": "Hugging Face TheKernel01/140k-Real-and-Fake-Faces",
                "domain": "FFHQ vs StyleGAN faces (256px)",
                "total_images": 140000,
                "train_set": 100000,
                "val_set": 20000,
                "test_set": 20000,
                "classes": ["Real (FFHQ Camera)", "Fake (StyleGAN)"],
            },
            "model_comparison": model_comparison,
            "confusion_matrices": confusion_matrices,
            "generator_breakdown": generator_breakdown,
            "raw_eval": data,
        }

    except Exception as e:
        logger.error(f"Failed to load benchmark data: {e}")
        return {"error": str(e)}
