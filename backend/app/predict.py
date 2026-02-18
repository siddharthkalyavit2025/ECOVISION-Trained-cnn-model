"""
Prediction logic — orchestrates preprocessing, inference, and result formatting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.config import settings
from app.model_loader import get_model
from app.utils import preprocess_image, generate_gradcam, log_prediction

logger = logging.getLogger(__name__)


def predict_image(
    image_bytes: bytes,
    filename: str = "unknown",
    return_gradcam: bool = False,
) -> dict[str, Any]:
    """Run inference on raw image bytes and return structured results.

    Args:
        image_bytes: Raw bytes of the uploaded image.
        filename: Original filename (used for logging).
        return_gradcam: Whether to include a Grad-CAM heatmap in the result.

    Returns:
        Dictionary containing:
            - predicted_class (str)
            - confidence (float)
            - all_probabilities (dict[str, float])
            - recyclability_suggestion (str)
            - below_confidence_threshold (bool)
            - gradcam_image (str, optional) — base64-encoded PNG
    """
    model = get_model()
    img_array = preprocess_image(image_bytes)

    # ── Inference ────────────────────────────────────────────────────────
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    predicted_class = settings.CLASS_LABELS[predicted_index]

    all_probabilities = {
        label: round(float(prob), 4)
        for label, prob in zip(settings.CLASS_LABELS, probabilities)
    }

    below_threshold = confidence < settings.CONFIDENCE_THRESHOLD

    result: dict[str, Any] = {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probabilities,
        "recyclability_suggestion": settings.RECYCLABILITY.get(
            predicted_class, "No suggestion available."
        ),
        "below_confidence_threshold": below_threshold,
    }

    if below_threshold:
        logger.warning(
            "Low confidence (%.2f) for '%s' on file '%s'.",
            confidence, predicted_class, filename,
        )

    # ── Optional Grad-CAM ────────────────────────────────────────────────
    if return_gradcam:
        try:
            gradcam_b64 = generate_gradcam(model, img_array, predicted_index)
            result["gradcam_image"] = gradcam_b64
        except Exception:
            logger.exception("Grad-CAM generation failed.")
            result["gradcam_image"] = ""

    # ── Log to CSV ───────────────────────────────────────────────────────
    log_prediction(filename, predicted_class, confidence)

    logger.info(
        "Prediction: %s (%.2f%%) for '%s'",
        predicted_class, confidence * 100, filename,
    )
    return result
