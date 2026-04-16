"""
FastAPI application — EcoVision Garbage Classification API.

Endpoints:
    GET  /              → Health check
    POST /predict       → Classify an uploaded image
    POST /predict/gradcam → Classify + return Grad-CAM heatmap
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model_loader import is_model_loaded, load_model
from app.predict import predict_image
from app.schemas import GradCAMResponse, HealthResponse, PredictionResponse
from app.servo import connect_arduino, disconnect_arduino, rotate_servo_for_class

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Allowed image MIME types ─────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


# ── Lifespan: load model at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model when the application starts."""
    logger.info("🚀 Starting EcoVision API …")
    try:
        load_model()
    except Exception as exc:
        logger.critical("Model failed to load: %s", exc)
        raise

    # Connect to Arduino (non-blocking — logs warning if unavailable)
    connect_arduino()

    yield

    disconnect_arduino()
    logger.info("👋 Shutting down EcoVision API.")


# ── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(
    title="EcoVision — Garbage Classification API",
    description=(
        "Upload an image of waste and receive a classification with confidence "
        "scores, recyclability suggestions, and optional Grad-CAM visualizations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health check",
    tags=["General"],
)
async def health_check() -> HealthResponse:
    """Return the API health status and whether the model is loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=is_model_loaded(),
        model_path=settings.MODEL_PATH,
    )



@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify an uploaded waste image",
    tags=["Prediction"],
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Accept an image file and return the predicted waste class.

    **Supported formats:** JPEG, PNG, WebP, BMP, TIFF.
    """
    _validate_upload(file)
    image_bytes = await file.read()
    result = predict_image(image_bytes, filename=file.filename or "upload")

    # ── Servo: rotate bin based on predicted class ────────────────
    servo_info = rotate_servo_for_class(result["predicted_class"])
    logger.info("Servo → %s", servo_info)
    result["bin_category"] = servo_info["bin_category"]
    result["servo_angle"] = servo_info["servo_angle"]

    return PredictionResponse(**result)


@app.post(
    "/predict/gradcam",
    response_model=GradCAMResponse,
    summary="Classify + Grad-CAM visualization",
    tags=["Prediction"],
)
async def predict_with_gradcam(
    file: UploadFile = File(...),
) -> GradCAMResponse:
    """Classify an uploaded image and return a Grad-CAM heatmap overlay."""
    _validate_upload(file)
    image_bytes = await file.read()
    result = predict_image(
        image_bytes, filename=file.filename or "upload", return_gradcam=True
    )

    # ── Servo: rotate bin based on predicted class ────────────────
    servo_info = rotate_servo_for_class(result["predicted_class"])
    logger.info("Servo → %s", servo_info)
    result["bin_category"] = servo_info["bin_category"]
    result["servo_angle"] = servo_info["servo_angle"]

    return GradCAMResponse(**result)


# ── Helpers ──────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile) -> None:
    """Validate that the uploaded file is an allowed image type."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file.content_type}. "
                f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )
