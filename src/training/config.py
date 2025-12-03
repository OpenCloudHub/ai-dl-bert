"""Training configuration for DistilBERT emotion classification."""

from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    """Configuration for training pipeline."""

    # MLflow Configuration
    mlflow_tracking_uri: str
    mlflow_experiment_name: str = "emotion-classification"
    mlflow_registered_model_name: str = "emotion-classifier"

    # DVC Configuration
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
        env_prefix = ""


# ============================================== #
# Data Contract - FROM ENV
# ============================================== #
class WorkflowTags(BaseSettings):
    argo_workflow_uid: str = "DEV"
    docker_image_tag: str = "DEV"
    dvc_data_version: str

    class Config:
        env_prefix = ""


WORKFLOW_TAGS = WorkflowTags()
TRAINING_CONFIG = TrainingConfig()
