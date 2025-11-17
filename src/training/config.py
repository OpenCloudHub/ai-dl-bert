"""Training configuration for DistilBERT emotion classification."""

from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    """Configuration for training pipeline."""

    # MLflow Configuration
    mlflow_tracking_uri: str = "http://mlflow.mlops.svc.cluster.local:5000"
    mlflow_experiment_name: str = "emotion-classification"
    mlflow_registered_model_name: str = "distilbert-emotion"

    # DVC Configuration
    dvc_repo: str = "https://github.com/OpenCloudHub/data-registry"
    dvc_train_data_path: str = "data/emotion/processed/train/train.parquet"
    dvc_val_data_path: str = "data/emotion/processed/val/val.parquet"
    dvc_metrics_path: str = "data/emotion/metadata.json"

    # Ray Configuration
    ray_storage_path: str = "/tmp/ray_results"

    # Model Configuration
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128

    class Config:
        env_prefix = ""


TRAINING_CONFIG = TrainingConfig()
