"""
Pydantic request / response schemas for the prediction API.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for the health-check endpoint."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    model_path: str = Field(..., examples=["garbage_efficientnet_model.h5"])


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    predicted_class: str = Field(..., examples=["plastic"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.92])
    all_probabilities: dict[str, float] = Field(
        ...,
        examples=[
            {
                "batteries": 0.01,
                "biological": 0.02,
                "cardboard": 0.01,
                "clothes": 0.01,
                "glass": 0.01,
                "metal": 0.01,
                "paper": 0.01,
                "plastic": 0.92,
                "shoes": 0.00,
                "trash": 0.00,
            }
        ],
    )
    recyclability_suggestion: str = Field(
        ...,
        examples=[
            "♻️ Recyclable — Check the resin code. Rinse and place in plastics recycling."
        ],
    )
    below_confidence_threshold: bool = Field(
        False,
        description="True when the top prediction confidence is below the configured threshold.",
    )


class GradCAMResponse(PredictionResponse):
    """Extended prediction response that includes a Grad-CAM heatmap."""

    gradcam_image: str = Field(
        ...,
        description="Base64-encoded PNG image of the Grad-CAM heatmap overlay.",
    )
