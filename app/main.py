import os
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.evaluation import get_benchmark_data
from inference.engine import analyze_with_trained_models, load_checkpoint

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "samples"
STATIC_DIR = BASE_DIR / "static"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"


def _load_all_checkpoints():
    """Load trained model checkpoints at startup."""
    # 1. Guided Model
    guided_ckpt = CHECKPOINTS_DIR / "guided" / "best.pt"
    if guided_ckpt.exists():
        load_checkpoint("guided", str(guided_ckpt), {
            "cnn_name": "efficientnet_b0",
            "vit_name": "vit_small_patch14_dinov2",
            "pretrained": False,
            "use_freq_stream": True,
            "use_spatial_gate": True,
        })

    # 2. CNN Baseline
    cnn_ckpt = CHECKPOINTS_DIR / "cnn_baseline" / "best.pt"
    if cnn_ckpt.exists():
        load_checkpoint("cnn", str(cnn_ckpt), {
            "backbone_name": "efficientnet_b0",
            "pretrained": False,
        })

    # 3. ViT Baseline
    vit_ckpt = CHECKPOINTS_DIR / "vit_baseline" / "best.pt"
    if vit_ckpt.exists():
        load_checkpoint("vit", str(vit_ckpt), {
            "backbone_name": "vit_small_patch14_dinov2",
            "pretrained": False,
        })

    # 4. Concat Hybrid
    hybrid_ckpt = CHECKPOINTS_DIR / "concat_hybrid" / "best.pt"
    if hybrid_ckpt.exists():
        load_checkpoint("hybrid", str(hybrid_ckpt), {
            "cnn_name": "efficientnet_b0",
            "vit_name": "vit_small_patch14_dinov2",
            "pretrained": False,
        })


@asynccontextmanager
async def lifespan(app_instance: "FastAPI"):
    _load_all_checkpoints()
    yield


app = FastAPI(
    title="DeepTruth AI — Fake / AI-Generated Image Detector",
    description="Intelligent Model Design Using AI: Real vs AI-Generated Face Detector (ViT-Guided CNN)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend assets and samples
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "DeepTruth AI API is running. Please check static/index.html"}


@app.get("/api/info")
def get_project_info():
    return {
        "title": "AI System for Detecting Fake / AI-Generated Images",
        "subject": "Intelligent Model Design Using AI",
        "institution": "Bennett University",
        "semester": "Semester 5 (IMD)",
        "team": [
            {"name": "Shashank Solaiappan", "id": "S24CSEU2692", "role": "Hybrid Model & Pipeline Architecture"},
            {"name": "Brajraj Mahajan", "id": "S24CSEU2543", "role": "CNN Baseline & Local Feature Extraction"},
            {"name": "Uma Shankar Mannuru", "id": "S24CSEU2552", "role": "Vision Transformer (ViT) & Patch Attention"},
            {"name": "Nidhi Verma", "id": "S24CSEU2553", "role": "Dataset Preprocessing, Augmentation & Metrics"}
        ],
        "models_evaluated": [
            "CNN Baseline (EfficientNet-B0)",
            "Vision Transformer (ViT-S/14 DINOv2)",
            "Concat-Hybrid (CNN + ViT)",
            "ViT-Guided CNN with Spatial Attention Gating & FiLM (Proposed)",
        ]
    }


@app.get("/api/samples")
def get_sample_images():
    """
    Returns curated authentic and AI-generated sample images for 1-click testing.
    """
    return [
        {
            "id": "stylegan_portrait",
            "filename": "stylegan_portrait.jpg",
            "url": "/samples/stylegan_portrait.jpg",
            "title": "Studio Fashion Portrait",
            "category": "AI-Generated",
            "generator": "StyleGAN2 Architecture",
            "description": "Synthetic face showing characteristic unnatural skin smoothness, asymmetric details, and checkerboard upsampling cues."
        },
        {
            "id": "midjourney_astronaut",
            "filename": "midjourney_astronaut.jpg",
            "url": "/samples/midjourney_astronaut.jpg",
            "title": "Cyberpunk Astronaut Portrait",
            "category": "AI-Generated",
            "generator": "Midjourney v6 / Latent Diffusion",
            "description": "High-fidelity synthetic portrait with glowing reflections and smooth neon shading."
        },
        {
            "id": "real_watchmaker",
            "filename": "real_watchmaker.jpg",
            "url": "/samples/real_watchmaker.jpg",
            "title": "Elderly Watchmaker at Work",
            "category": "Authentic Photograph",
            "generator": "Real Camera DSLR (50mm f/1.8)",
            "description": "Authentic photograph with natural optical depth-of-field and lens grain noise consistency."
        },
        {
            "id": "real_lake_mountain",
            "filename": "real_lake_mountain.jpg",
            "url": "/samples/real_lake_mountain.jpg",
            "title": "Alpine Lake & Glacial Peaks",
            "category": "Authentic Photograph",
            "generator": "Real Camera DSLR (Landscape)",
            "description": "Natural outdoor photography with coherent reflection physics and organic texture gradients."
        }
    ]


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(None),
    sample_id: str = Form(None)
):
    """
    Accepts either an uploaded image file or a preloaded sample_id,
    and runs inference through the trained models (or heuristic fallback if no ckpt).
    """
    image_bytes = None
    filename = "custom_upload.jpg"

    if sample_id:
        sample_path = SAMPLES_DIR / f"{sample_id}.jpg"
        if not sample_path.exists():
            matches = list(SAMPLES_DIR.glob(f"{sample_id}*"))
            if matches:
                sample_path = matches[0]
            else:
                raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        try:
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
            filename = sample_path.name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read sample image: {str(e)}")

    elif file:
        filename = file.filename or "upload.jpg"
        # Validate file size (< 25MB)
        image_bytes = await file.read()
        if len(image_bytes) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 25MB limit.")
    else:
        raise HTTPException(status_code=400, detail="Please upload an image file or specify a sample_id.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    # Execute DeepTruth AI pipeline (trained models or fallback)
    result = analyze_with_trained_models(image, filename=filename)
    result["filename"] = filename
    return result


@app.get("/api/benchmark")
def get_benchmarks():
    return get_benchmark_data()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
