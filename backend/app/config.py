"""
Application configuration module.

Loads settings from environment variables with sensible defaults.
Uses Pydantic BaseSettings for validation and .env file support.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Model ────────────────────────────────────────────────────────────
    MODEL_PATH: str = str(
        Path(__file__).resolve().parent.parent.parent / "garbage_efficientnet_model.h5"
    )
    IMAGE_SIZE: int = 224

    # ── Server ───────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: list[str] = ["*"]

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── Prediction ───────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.5
    PREDICTION_LOG_PATH: str = "predictions.csv"

    # ── Class labels (alphabetical – matches ImageDataGenerator default) ─
    CLASS_LABELS: list[str] = [
        "batteries",
        "biological",
        "cardboard",
        "clothes",
        "glass",
        "metal",
        "paper",
        "plastic",
        "shoes",
        "trash",
    ]

    # ── Recyclability mapping ────────────────────────────────────────────
    RECYCLABILITY: dict[str, str] = {
        "batteries": "♻️ Recyclable — Take to a battery recycling drop-off point. Never throw in regular trash.",
        "biological": "🌿 Compostable — Can be composted. Use a compost bin or green waste collection.",
        "cardboard": "♻️ Recyclable — Flatten and place in the recycling bin. Remove any tape or staples.",
        "clothes": "👗 Reusable / Donatable — Donate to charity or textile recycling programs.",
        "glass": "♻️ Recyclable — Rinse and place in glass recycling. Separate by color if required.",
        "metal": "♻️ Recyclable — Rinse cans/tins and place in recycling. Includes aluminium & steel.",
        "paper": "♻️ Recyclable — Place in paper recycling. Avoid wet or food-stained paper.",
        "plastic": "♻️ Recyclable — Check the resin code. Rinse and place in plastics recycling.",
        "shoes": "👟 Reusable / Donatable — Donate wearable shoes. Some brands offer take-back programs.",
        "trash": "🗑️ Non-Recyclable — Dispose in general waste. Consider reducing consumption of such items.",
    }

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton settings instance
settings = Settings()
