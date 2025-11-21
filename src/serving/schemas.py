"""Schema definitions for the Fashion MNIST serving module."""

from datetime import datetime
from enum import StrEnum, auto
from typing import List

from pydantic import BaseModel, Field

from src._utils.logging import get_logger
from src.serving.config import SERVING_CONFIG

logger = get_logger(__name__)


class APIStatus(StrEnum):
    """API status enumeration."""

    LOADING = auto()
    HEALTHY = auto()
    UNHEALTHY = auto()
    NOT_READY = auto()


class PredictionRequest(BaseModel):
    """Input model for predictions."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=SERVING_CONFIG.request_max_length,
        description="List of text strings to classify",
        examples=[["I love this!", "I'm so sad"]],
    )


class Prediction(BaseModel):
    """Single prediction result."""

    text: str = Field(..., description="Input text")
    emotion: str = Field(..., description="Predicted emotion label")
    confidence: float = Field(
        ..., description="Prediction confidence (0-1)", ge=0.0, le=1.0
    )
    all_scores: dict[str, float] = Field(
        ..., description="Confidence scores for all emotion classes"
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predictions: List[Prediction] = Field(
        ..., description="List of predictions for each input text"
    )
    model_uri: str = Field(..., description="URI of the model used")
    timestamp: datetime = Field(..., description="Prediction timestamp UTC")
    processing_time_ms: float = Field(
        ..., description="Time taken to process request in milliseconds"
    )


class ModelInfo(BaseModel):
    """Model metadata information."""

    model_uri: str = Field(..., description="URI of the model used")
    model_uuid: str = Field(..., description="MLflow model UUID")
    run_id: str = Field(..., description="MLflow run ID associated with the model")
    model_signature: dict | None = Field(None, description="MLflow model signature")
    data_version: str | None = Field(
        None, description="DVC data version used for training"
    )
    training_timestamp: datetime | None = Field(
        None, description="When the model was trained"
    )
    emotion_labels: List[str] = Field(
        default_factory=list, description="List of emotion class labels"
    )
    api_name: str = Field(..., description="API name")
    max_length: int = Field(..., description="Maximum sequence length")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: APIStatus = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    model_uri: str | None = Field(None, description="Current model URI")
    uptime_seconds: int | None = Field(None, description="Service uptime in seconds")


class RootResponse(BaseModel):
    """Response model for root endpoint."""

    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Service status")
    docs: str = Field(..., description="URL to API documentation")
    health: str = Field(..., description="URL to health check endpoint")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error details")
