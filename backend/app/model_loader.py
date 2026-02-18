"""
Model loading module.

Loads the trained EfficientNet Keras model once and exposes it as a
singleton so every request re-uses the same in-memory model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import tensorflow as tf

from app.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singleton ──────────────────────────────────────────────────
_model: Optional[tf.keras.Model] = None


def load_model() -> tf.keras.Model:
    """Load the Keras model from disk and cache it.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If loading fails for any other reason.
    """
    global _model

    if _model is not None:
        logger.info("Model already loaded — skipping reload.")
        return _model

    model_path = Path(settings.MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("Loading model from %s …", model_path)
    logger.info(
        "Available devices: %s",
        [d.name for d in tf.config.list_physical_devices()],
    )

    try:
        _model = tf.keras.models.load_model(str(model_path))
        logger.info("✅ Model loaded successfully.")
        return _model
    except Exception as exc:
        logger.exception("Failed to load model.")
        raise RuntimeError(f"Model loading failed: {exc}") from exc


def get_model() -> tf.keras.Model:
    """Return the cached model, loading it first if necessary."""
    if _model is None:
        return load_model()
    return _model


def is_model_loaded() -> bool:
    """Check whether the model has been loaded into memory."""
    return _model is not None
