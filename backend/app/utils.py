"""
Utility helpers for image preprocessing, Grad-CAM, and prediction logging.
"""

from __future__ import annotations

import base64
import csv
import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Image preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes into a preprocessed tensor ready for inference.

    Steps:
        1. Open image with Pillow and convert to RGB.
        2. Resize to (IMAGE_SIZE × IMAGE_SIZE).
        3. Convert to float32 numpy array.
        4. Normalize pixel values to [0, 1].
        5. Expand to batch dimension → shape (1, H, W, 3).

    Args:
        image_bytes: Raw bytes of the uploaded image.

    Returns:
        Preprocessed numpy array with shape (1, 224, 224, 3).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    logger.debug("Preprocessed image shape: %s", img_array.shape)
    return img_array


# ═══════════════════════════════════════════════════════════════════════════
#  Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════

def generate_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    predicted_index: int,
    layer_name: str | None = None,
) -> str:
    """Generate a Grad-CAM heatmap and return it as a base64-encoded PNG.

    Args:
        model: The loaded Keras model.
        img_array: Preprocessed image array (batch of 1).
        predicted_index: Index of the predicted class.
        layer_name: Name of the target convolutional layer. If ``None``,
            the last Conv2D layer is used automatically.

    Returns:
        Base64-encoded PNG string of the heatmap overlay.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Find the last Conv2D layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
        if layer_name is None:
            # Try to find conv layers inside nested models (e.g. EfficientNet)
            for layer in model.layers:
                if hasattr(layer, "layers"):
                    for sub_layer in reversed(layer.layers):
                        if isinstance(sub_layer, tf.keras.layers.Conv2D):
                            layer_name = sub_layer.name
                            # Build a grad model from the inner model
                            inner_model = layer
                            grad_model = tf.keras.models.Model(
                                inputs=inner_model.input,
                                outputs=[
                                    inner_model.get_layer(layer_name).output,
                                    inner_model.output,
                                ],
                            )
                            return _compute_gradcam(
                                grad_model, model, img_array,
                                predicted_index, use_inner=True,
                            )

    # Standard path
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    return _compute_gradcam(grad_model, model, img_array, predicted_index)


def _compute_gradcam(
    grad_model: tf.keras.Model,
    full_model: tf.keras.Model,
    img_array: np.ndarray,
    predicted_index: int,
    use_inner: bool = False,
) -> str:
    """Internal: compute Grad-CAM and render overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    with tf.GradientTape() as tape:
        if use_inner:
            # For nested EfficientNet: get inner model's output
            inner_input = full_model.layers[0] if hasattr(full_model.layers[0], "input") else None
            conv_output, predictions = grad_model(img_array)
            # Use the full model for final predictions
            final_preds = full_model(img_array)
            loss = final_preds[:, predicted_index]
        else:
            conv_output, predictions = grad_model(img_array)
            loss = predictions[:, predicted_index]
        grads = tape.gradient(loss, conv_output)

    if grads is None:
        logger.warning("Grad-CAM: gradients are None – returning blank overlay.")
        return _blank_overlay(img_array)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to match original image
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap_resized).resize(
        (settings.IMAGE_SIZE, settings.IMAGE_SIZE)
    )
    heatmap_array = np.array(heatmap_img) / 255.0

    # Apply colormap
    colormap = cm.jet(heatmap_array)[:, :, :3]

    # Overlay on original image
    original = img_array[0]
    overlay = 0.6 * original + 0.4 * colormap

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(overlay)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _blank_overlay(img_array: np.ndarray) -> str:
    """Return a base64 PNG of the original image when Grad-CAM fails."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img_array[0])
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
#  Prediction CSV logger
# ═══════════════════════════════════════════════════════════════════════════

def log_prediction(
    filename: str,
    predicted_class: str,
    confidence: float,
) -> None:
    """Append a prediction record to the CSV log file.

    Args:
        filename: Original name of the uploaded file.
        predicted_class: Name of the predicted class.
        confidence: Confidence score for the top prediction.
    """
    log_path = Path(settings.PREDICTION_LOG_PATH)
    file_exists = log_path.exists()

    try:
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "filename", "predicted_class", "confidence"])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                filename,
                predicted_class,
                f"{confidence:.4f}",
            ])
    except OSError:
        logger.warning("Could not write to prediction log at %s", log_path)
