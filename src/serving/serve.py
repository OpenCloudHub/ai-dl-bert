# =============================================================================
# serve.py - Ray Serve + FastAPI Model Serving
# =============================================================================
#
# Purpose:
#   Production-ready model serving using Ray Serve with FastAPI integration.
#   Loads models from MLflow Registry and supports hot updates without restart.
#
# Why Ray Serve + FastAPI?
#   - Ray Serve: Distributed serving with auto-scaling
#   - FastAPI: Async API with OpenAPI docs
#   - MLflow: Load models by URI (models:/name/version)
#   - Hot Reload: reconfigure() for zero-downtime updates
#
# Endpoints:
#   GET  /         - Service info
#   GET  /health   - Health check with model status
#   GET  /info     - Model metadata (version, labels, training info)
#   POST /predict  - Batch emotion prediction
#
# Usage:
#   serve run src.serving.serve:app_builder model_uri="models:/name/1" --reload
#
# =============================================================================
"""
Ray Serve + FastAPI serving application for emotion classification.

Provides model loading from MLflow, batch prediction, health checks,
and hot model updates via reconfigure().
"""
- Hot model updates via reconfigure() method
"""

from datetime import datetime, timezone

import mlflow
import torch
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from ray import serve
from ray.serve import Application
from transformers import AutoTokenizer

from src._utils.logging import get_logger
from src.serving.config import SERVING_CONFIG
from src.serving.schemas import (
    APIStatus,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
)

logger = get_logger(__name__)

# FastAPI application with OpenAPI documentation
app = FastAPI(
    title="😊 Emotion Classification API",
    description="Emotion classification using Ray Serve + MLflow + DistilBERT",
    version="1.0.0",
)


@serve.deployment(
    ray_actor_options={
        "num_cpus": 2,
        "num_gpus": 1 if torch.cuda.is_available() else 0,
    },
)
@serve.ingress(app)
class EmotionClassifier:
    """
    Ray Serve deployment for emotion classification.

    This class wraps a DistilBERT model loaded from MLflow and exposes
    it via FastAPI endpoints. It supports dynamic model updates through
    the reconfigure() method, enabling zero-downtime model updates.

    Attributes:
        status: Current API health status (LOADING, HEALTHY, UNHEALTHY)
        device: PyTorch device (cuda/cpu) for inference
        model: Loaded PyTorch Lightning model
        tokenizer: HuggingFace tokenizer for input processing
        model_info: ModelInfo with version and training metadata
        label2id: Mapping from emotion labels to class indices
        id2label: Mapping from class indices to emotion labels
        start_time: Service start time for uptime tracking
    """

    def __init__(self, model_uri: str | None = None) -> None:
        """
        Initialize the emotion classifier deployment.

        Args:
            model_uri: Optional MLflow model URI to load at startup.
                      Format: "models:/model-name/version" or "runs:/run-id/model"
                      If not provided, deployment starts in NOT_READY state.
        """
        logger.info("😊 Initializing Emotion Classification Service")
        self.status = APIStatus.NOT_READY
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.model_info: ModelInfo | None = None
        self.data_version: str | None = None
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}
        self.start_time = datetime.now(timezone.utc)

        # Load model if URI provided at init
        if model_uri:
            try:
                self._load_model(model_uri)
            except Exception as e:
                logger.error(f"Failed to load model during initialization: {e}")
                self.status = APIStatus.UNHEALTHY

    def _load_model(self, model_uri: str) -> None:
        """
        Load model from MLflow and initialize tokenizer.

        Fetches the model from MLflow Model Registry, loads label mappings
        from training artifacts, and initializes the HuggingFace tokenizer.

        Args:
            model_uri: MLflow model URI (e.g., "models:/emotion-classifier/1")

        Raises:
            HTTPException: If model loading fails (503 for MLflow errors,
                          500 for unexpected errors)
        """
        logger.info(f"📦 Loading model from: {model_uri}")
        self.status = APIStatus.LOADING

        try:
            # Get model info first to validate URI
            info = mlflow.models.get_model_info(model_uri)

            # Get training run metadata for data lineage
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(info.run_id)

            # Extract DVC version from run tags for data lineage tracking
            self.data_version = run.data.tags.get("dvc_data_version")
            if not self.data_version:
                logger.warning("No dvc_data_version found in run tags")

            logger.info(f"📊 Data version: {self.data_version}")

            # Load the PyTorch model (Lightning module saved by mlflow.pytorch)
            self.model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
            self.model.eval()  # Set to evaluation mode

            # Load tokenizer (must match training tokenizer)
            logger.info(f"🔤 Loading tokenizer: {SERVING_CONFIG.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(SERVING_CONFIG.model_name)

            # Load label mappings from training artifacts
            try:
                labels_artifact = client.download_artifacts(
                    run.info.run_id, "labels.json"
                )
                import json

                with open(labels_artifact, "r") as f:
                    labels_data = json.load(f)
                    self.label2id = labels_data.get("label2id", {})
                    # Convert string keys to int for id2label (JSON limitation)
                    id2label_raw = labels_data.get("id2label", {})
                    self.id2label = {int(k): v for k, v in id2label_raw.items()}
                logger.info(
                    f"✅ Loaded {len(self.label2id)} emotion labels from artifacts"
                )
            except Exception as e:
                logger.warning(f"⚠️  Could not load label mappings from artifacts: {e}")
                # Fallback: generate generic labels based on model config
                num_labels = self.model.model.config.num_labels
                self.id2label = {i: f"emotion_{i}" for i in range(num_labels)}
                self.label2id = {v: k for k, v in self.id2label.items()}

            # Extract training timestamp from MLflow run
            training_timestamp = datetime.fromtimestamp(
                run.info.start_time / 1000.0, tz=timezone.utc
            )

            # Build ModelInfo for /info endpoint
            self.model_info = ModelInfo(
                model_uri=model_uri,
                model_uuid=info.model_uuid,
                run_id=info.run_id,
                model_signature=info.signature.to_dict() if info.signature else None,
                data_version=self.data_version,
                training_timestamp=training_timestamp,
                emotion_labels=list(self.id2label.values()),
                api_name=SERVING_CONFIG.api_name,
                max_length=SERVING_CONFIG.request_max_length,
            )

            self.status = APIStatus.HEALTHY
            logger.success("✅ Model loaded successfully")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Model UUID: {self.model_info.model_uuid}")
            logger.info(f"   Run ID: {self.model_info.run_id}")
            logger.info(f"   Emotion labels: {list(self.id2label.values())}")

        except mlflow.exceptions.MlflowException as e:
            self.status = APIStatus.UNHEALTHY
            logger.error(f"❌ MLflow error loading model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load model from MLflow: {str(e)}",
            )
        except Exception as e:
            self.status = APIStatus.UNHEALTHY
            logger.error(f"❌ Unexpected error loading model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error loading model: {str(e)}",
            )

    def reconfigure(self, config: dict) -> None:
        """
        Handle dynamic model updates without service restart.

        This method is called by Ray Serve when user_config changes,
        enabling hot model swaps. See Ray Serve documentation:
        https://docs.ray.io/en/latest/serve/advanced-guides/inplace-updates.html

        Args:
            config: Dict containing new configuration, expects "model_uri" key

        Usage:
            serve.run(..., user_config={"model_uri": "models:/model/2"})
        """
        new_model_uri = config.get("model_uri")

        if not new_model_uri:
            logger.warning("⚠️ No model_uri provided in config")
            return

        # If no model loaded yet, load it
        if self.model_info is None:
            logger.info("🆕 Initial model load via reconfigure")
            self._load_model(new_model_uri)
            return

        # Check if URI changed before reloading
        if self.model_info.model_uri != new_model_uri:
            logger.info(
                f"🔄 Updating model from {self.model_info.model_uri} to {new_model_uri}"
            )
            self._load_model(new_model_uri)
        else:
            logger.info("ℹ️ Model URI unchanged, skipping reload")

    # =========================================================================
    # API Endpoints
    # =========================================================================

    @app.get(
        "/",
        response_model=RootResponse,
        summary="Root endpoint",
        responses={
            200: {"description": "Service information"},
            503: {"description": "Service not healthy"},
        },
    )
    async def root(self):
        """
        Root endpoint providing service info and documentation links.

        Returns basic service information including name, version, status,
        and links to documentation and health check endpoints.
        """
        return RootResponse(
            service="Emotion Classification API",
            version="1.0.0",
            status=self.status.value,
            docs="/docs",
            health="/health",
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Health Check",
        responses={
            200: {"description": "Service is healthy"},
            503: {"description": "Service is not ready or unhealthy"},
        },
    )
    async def health(self):
        """
        Health check endpoint for load balancers and orchestration.

        Returns current service status, model loading state, and uptime.
        Returns HTTP 503 if service is not healthy (useful for Kubernetes
        readiness probes).
        """
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        response = HealthResponse(
            status=self.status,
            model_loaded=self.model is not None,
            model_uri=self.model_info.model_uri if self.model_info else None,
            uptime_seconds=int(uptime),
        )

        # Return 503 if not healthy (for load balancer health checks)
        if self.status != APIStatus.HEALTHY:
            detail = response.model_dump()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=detail,
            )
        return response

    @app.get(
        "/info",
        response_model=ModelInfo,
        summary="Model Information",
        responses={
            200: {"description": "Model information"},
            503: {"description": "Model not loaded", "model": ErrorResponse},
        },
    )
    async def info(self):
        """
        Get detailed model metadata including training information.

        Returns comprehensive model information including:
        - Model URI and version
        - MLflow run ID and model UUID
        - Training data version (DVC)
        - Available emotion labels
        - Training timestamp
        """
        if self.model_info is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please configure the deployment with a model_uri.",
            )
        return self.model_info

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Predict Emotion",
        responses={
            200: {"description": "Successful prediction"},
            400: {"description": "Invalid input", "model": ErrorResponse},
            503: {"description": "Model not loaded", "model": ErrorResponse},
            500: {"description": "Internal server error", "model": ErrorResponse},
        },
    )
    async def predict(self, request: PredictionRequest):
        """
        Predict emotions from text inputs (batch processing supported).

        Tokenizes input texts, runs inference through DistilBERT, and returns
        predictions with confidence scores for all emotion classes.

        **Input Format:**
        - List of text strings (1-100 texts per request)
        - Text will be automatically tokenized and truncated to max_length

        **Output:**
        - emotion: Predicted emotion label (e.g., 'joy', 'sadness', 'anger')
        - confidence: Softmax probability of the predicted class (0-1)
        - all_scores: Dictionary with confidence scores for all emotion classes

        **Example:**
        ```json
        {
            "texts": ["I love this!", "I'm so sad today"]
        }
        ```
        """
        # Check if model is loaded
        if self.model is None or self.model_info is None or self.tokenizer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Configure the deployment with a model_uri.",
            )

        start_time = datetime.now(timezone.utc)

        try:
            # Tokenize inputs
            encoded = self.tokenizer(
                request.texts,
                padding="max_length",
                truncation=True,
                max_length=SERVING_CONFIG.request_max_length,
                return_tensors="pt",
            )

            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                confidences, class_ids = torch.max(probs, dim=1)

            # Build predictions
            predictions = []
            for idx, (text, cls_id, conf, prob_scores) in enumerate(
                zip(
                    request.texts,
                    class_ids.cpu().numpy(),
                    confidences.cpu().numpy(),
                    probs.cpu().numpy(),
                )
            ):
                # Create all_scores dict
                all_scores = {
                    self.id2label[i]: float(score)
                    for i, score in enumerate(prob_scores)
                }

                predictions.append(
                    Prediction(
                        text=text,
                        emotion=self.id2label[int(cls_id)],
                        confidence=float(conf),
                        all_scores=all_scores,
                    )
                )

            # Calculate processing time
            processing_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            return PredictionResponse(
                predictions=predictions,
                model_uri=self.model_info.model_uri,
                timestamp=datetime.now(timezone.utc),
                processing_time_ms=processing_time,
            )

        except HTTPException:
            raise
        except ValueError as e:
            logger.error(f"❌ Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input: {str(e)}",
            )
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}",
            )


class AppBuilderArgs(BaseModel):
    """Arguments for building the Ray Serve application."""

    model_uri: str | None = Field(
        None,
        description="MLflow model URI to load (e.g., models:/ci.emotion-classifier/1 or runs:/run_id/model)",
    )


def app_builder(args: AppBuilderArgs) -> Application:
    """Helper function to build the deployment with optional model URI.

    Examples:
        Basic usage:
        >>> serve run src.serving.serve:app_builder model_uri="models:/ci.emotion-classifier/1"

        With hot reload for development:
        >>> serve run src.serving.serve:app_builder model_uri="models:/ci.emotion-classifier/1" --reload

    Args:
        args: Configuration arguments including model URI

    Returns:
        Ray Serve Application ready to deploy
    """
    return EmotionClassifier.bind(model_uri=args.model_uri)
