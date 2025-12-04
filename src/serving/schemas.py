# =============================================================================
# schemas.py - Pydantic Models for API Request/Response Validation
# =============================================================================
#
# Purpose:
#   Defines Pydantic models for API request validation and response
#   serialization. These schemas provide automatic validation, OpenAPI
#   documentation generation, and type safety for the FastAPI endpoints.
#
# Schema Categories:
#   1. Request Models: Input validation for API endpoints
#      - PredictionRequest: Batch text input for emotion prediction
#
#   2. Response Models: Output serialization for API responses
#      - PredictionResponse: Batch predictions with confidence scores
#      - ModelInfo: Model metadata for /info endpoint
#      - HealthResponse: Service health status
#      - RootResponse: Service info for root endpoint
#
#   3. Utility Models: Supporting types
#      - APIStatus: Enum for service health states
#      - Prediction: Single prediction result
#      - ErrorResponse: Standardized error format
#
# =============================================================================
"""
Pydantic schema definitions for the Emotion Classification API.

Provides request/response models with:
- Automatic validation and error messages
- OpenAPI documentation generation
- Type-safe serialization
"""

from datetime import datetime
from enum import StrEnum, auto
from typing import List

from pydantic import BaseModel, Field

from src._utils.logging import get_logger
from src.serving.config import SERVING_CONFIG

logger = get_logger(__name__)


# =============================================================================
# API Status Enumeration
# =============================================================================
class APIStatus(StrEnum):
    """
    API health status enumeration.

    Used by health check endpoint to report service state:
    - LOADING: Model is being loaded
    - HEALTHY: Service ready to accept requests
    - UNHEALTHY: Service encountered an error
    - NOT_READY: No model loaded yet
    """

    LOADING = auto()
    HEALTHY = auto()
    UNHEALTHY = auto()
    NOT_READY = auto()


# =============================================================================
# Request Models
# =============================================================================
class PredictionRequest(BaseModel):
    """
    Input model for batch emotion prediction.

    Accepts a list of text strings to classify. Each text will be
    tokenized and processed independently, returning predictions
    for all texts in a single response.

    Attributes:
        texts: List of text strings (1 to max_length items)

    Example:
        {"texts": ["I love this!", "I'm so sad"]}
    """

    texts: List[str] = Field(
        ...,  # Required field
        min_length=1,
        max_length=SERVING_CONFIG.request_max_length,
        description="List of text strings to classify",
        examples=[["I love this!", "I'm so sad"]],
    )


# =============================================================================
# Response Models - Predictions
# =============================================================================
class Prediction(BaseModel):
    """
    Single prediction result for one input text.

    Contains the predicted emotion, confidence score, and full
    probability distribution across all emotion classes.

    Attributes:
        text: Original input text
        emotion: Predicted emotion label (e.g., "joy", "sadness")
        confidence: Softmax probability of predicted class (0-1)
        all_scores: Dict of all emotion classes with their scores
    """

    text: str = Field(..., description="Input text")
    emotion: str = Field(..., description="Predicted emotion label")
    confidence: float = Field(
        ..., description="Prediction confidence (0-1)", ge=0.0, le=1.0
    )
    all_scores: dict[str, float] = Field(
        ..., description="Confidence scores for all emotion classes"
    )


class PredictionResponse(BaseModel):
    """
    Response model for batch prediction endpoint.

    Contains predictions for all input texts along with metadata
    about the model used and processing time.

    Attributes:
        predictions: List of Prediction objects for each input
        model_uri: MLflow URI of the model used
        timestamp: UTC timestamp of the prediction
        processing_time_ms: Inference time in milliseconds
    """

    predictions: List[Prediction] = Field(
        ..., description="List of predictions for each input text"
    )
    model_uri: str = Field(..., description="URI of the model used")
    timestamp: datetime = Field(..., description="Prediction timestamp UTC")
    processing_time_ms: float = Field(
        ..., description="Time taken to process request in milliseconds"
    )


# =============================================================================
# Response Models - Model Info
# =============================================================================
class ModelInfo(BaseModel):
    """
    Comprehensive model metadata for /info endpoint.

    Provides full transparency about the deployed model including
    version, training data, and available emotion labels.

    Attributes:
        model_uri: MLflow model URI (e.g., "models:/name/1")
        model_uuid: Unique MLflow model identifier
        run_id: MLflow run ID from training
        model_signature: Input/output schema if available
        data_version: DVC tag used for training data
        training_timestamp: When the model was trained
        emotion_labels: List of emotion classes the model predicts
        api_name: Name of this API service
        max_length: Maximum input sequence length
    """

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


# =============================================================================
# Response Models - Health & Status
# =============================================================================
class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.

    Used by load balancers and Kubernetes probes to determine
    service availability.

    Attributes:
        status: Current API status (HEALTHY, LOADING, etc.)
        model_loaded: Whether a model is currently loaded
        model_uri: URI of loaded model (if any)
        uptime_seconds: Service uptime in seconds
    """

    status: APIStatus = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether a model is loaded")
    model_uri: str | None = Field(None, description="Current model URI")
    uptime_seconds: int | None = Field(None, description="Service uptime in seconds")


class RootResponse(BaseModel):
    """
    Response model for root endpoint.

    Provides basic service information and links to documentation.
    """

    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Service status")
    docs: str = Field(..., description="URL to API documentation")
    health: str = Field(..., description="URL to health check endpoint")


class ErrorResponse(BaseModel):
    """
    Standardized error response model.

    Used for consistent error formatting across all endpoints.
    """

    detail: str = Field(..., description="Error details")
