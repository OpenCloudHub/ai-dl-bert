# =============================================================================
# config.py - Training Pipeline Configuration
# =============================================================================
#
# Purpose:
#   Centralized configuration management for the training pipeline using
#   Pydantic Settings. This module provides type-safe configuration loaded
#   from environment variables, enabling the same code to run in development
#   (with .env files) and production (with Kubernetes secrets/configmaps).
#
# Configuration Categories:
#   1. TrainingConfig: Static settings for training infrastructure
#      - MLflow tracking server URL and experiment names
#      - DVC repository and data paths
#      - Model and tokenization settings
#
#   2. WorkflowTags: Dynamic tags passed from CI/CD pipelines
#      - Argo workflow ID for traceability
#      - Docker image tag for reproducibility
#      - DVC data version for data lineage
#
# Usage:
#   from src.training.config import TRAINING_CONFIG, WORKFLOW_TAGS
#
#   # Access configuration values
#   mlflow.set_tracking_uri(TRAINING_CONFIG.mlflow_tracking_uri)
#   mlflow.set_tag("dvc_version", WORKFLOW_TAGS.dvc_data_version)
#
# Environment Variables:
#   See TrainingConfig and WorkflowTags classes for required variables.
#
# =============================================================================
"""
Training configuration using Pydantic Settings.

Provides type-safe configuration loaded from environment variables:
- TrainingConfig: MLflow, DVC, Ray, and model settings
- WorkflowTags: CI/CD metadata for experiment tracking
"""

from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    """
    Configuration for training pipeline infrastructure.

    Loaded from environment variables with optional defaults.
    Required variables must be set in environment or .env file.

    Attributes:
        mlflow_tracking_uri: URL of MLflow tracking server (REQUIRED)
        mlflow_experiment_name: Experiment name in MLflow UI
        mlflow_registered_model_name: Name for model registry
        dvc_repo: Git URL of DVC data registry
        dvc_train_data_path: Path to training data in registry
        dvc_val_data_path: Path to validation data in registry
        dvc_metrics_path: Path to metadata JSON in registry
        dvc_remote: Optional DVC remote name
        ray_storage_path: Local path for Ray Tune results
        model_name: HuggingFace model identifier
        max_length: Maximum sequence length for tokenization
    """

    # MLflow Configuration (tracking server and experiment settings)
    mlflow_tracking_uri: str  # Required: e.g., "http://mlflow:5000"
    mlflow_experiment_name: str = "emotion-classification"
    mlflow_registered_model_name: str = "emotion-classifier"

    # DVC Configuration (data registry settings)
    dvc_repo: str = "https://github.com/OpenCloudHub/data-registry"
    dvc_train_data_path: str = "data/emotion/processed/train/train.parquet"
    dvc_val_data_path: str = "data/emotion/processed/val/val.parquet"
    dvc_metrics_path: str = "data/emotion/metadata.json"
    dvc_remote: str | None = None

    # Ray Configuration
    ray_storage_path: str = "/tmp/ray_results"

    # Model Configuration
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128

    class Config:
        env_prefix = ""  # No prefix, use variable names directly


# =============================================================================
# Workflow Tags - CI/CD Metadata
# =============================================================================
class WorkflowTags(BaseSettings):
    """
    Dynamic tags from CI/CD pipeline for experiment tracking.

    These tags are passed from Argo Workflows or GitHub Actions to
    enable traceability between MLflow experiments and the CI/CD
    pipeline that triggered them.

    Attributes:
        argo_workflow_uid: Argo workflow unique ID (for Kubernetes runs)
        docker_image_tag: Docker image tag used for training
        dvc_data_version: Git tag from DVC registry (REQUIRED)
    """

    argo_workflow_uid: str = "DEV"  # Argo workflow ID, "DEV" for local runs
    docker_image_tag: str = "DEV"  # Docker tag, "DEV" for local runs
    dvc_data_version: str  # Required: e.g., "emotion-v0.3.0"

    class Config:
        env_prefix = ""


# Singleton instances for import
WORKFLOW_TAGS = WorkflowTags()
TRAINING_CONFIG = TrainingConfig()
