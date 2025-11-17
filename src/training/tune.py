"""Training script with Ray Tune hyperparameter optimization."""

import lightning as L
import mlflow
import torch
from pydantic_settings import BaseSettings
from ray import tune
from ray.air.integrations.mlflow import setup_mlflow
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from src._utils.logging import get_logger, log_section
from src.training.config import TRAINING_CONFIG
from src.training.data import load_data
from src.training.model import DistilBERTClassificationModule

logger = get_logger(__name__)


# ============================================== #
# 🔹 SECTION: Data Contract
# ============================================== #
class WorkflowTags(BaseSettings):
    """⚠️ Data Contract for CI/CD Workflows."""

    argo_workflow_uid: str
    docker_image_tag: str
    dvc_data_version: str


WORKFLOW_TAGS = WorkflowTags()


# ============================================== #
# 🔹 SECTION: Training Function
# ============================================== #
def train_emotion_classifier(
    config: dict,
    parent_run_id: str,
    train_ds,
    val_ds,
    num_classes: int,
    num_epochs: int = 1,  # Reduced to 1 for faster demo
):
    """Training function for a single trial"""
    # Ray's MLflow integration - handles run creation automatically
    # TODO: Wait for fix: https://github.com/ray-project/ray/pull/58705
    mlflow = setup_mlflow(
        config=config,
        tracking_uri="your_mlflow_uri",  # Or set via environment
        experiment_name="emotion_classification",
        rank_zero_only=True,
    )

    # Enable autolog
    mlflow.pytorch.autolog(log_models=False)

    # Create DataLoaders
    train_dataloader = train_ds.iter_torch_batches(batch_size=config["batch_size"])
    val_dataloader = val_ds.iter_torch_batches(batch_size=config["batch_size"])

    # Initialize model
    model = DistilBERTClassificationModule(
        model_name=TRAINING_CONFIG.model_name,
        num_classes=num_classes,
        learning_rate=config["learning_rate"],
    )

    # Configure Lightning trainer with Tune callback
    trainer = L.Trainer(
        max_epochs=num_epochs,
        devices="auto",
        accelerator="auto",
        callbacks=[
            TuneReportCallback(["val_loss", "val_acc", "val_f1"], on="validation_end")
        ],
        enable_progress_bar=False,
        enable_checkpointing=True,  # Enable checkpointing to save best model
        default_root_dir="/tmp/lightning_checkpoints",
    )

    # # Setup MLflow run (WITHOUT autolog to avoid overhead)
    # with mlflow.start_run(
    #     run_name=f"trial_lr{config['learning_rate']:.2e}_bs{config['batch_size']}",
    #     nested=True,
    #     parent_run_id=parent_run_id,
    #     tags=WORKFLOW_TAGS.model_dump(),
    # ) as child_run:
    #     logger.info(f"Started MLflow run {child_run.info.run_id} for trial")

    #     # Enable PyTorch Lightning autologging (without model logging)
    #     mlflow.pytorch.autolog(log_models=False)

    # Train
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


# ============================================== #
# 🔹 SECTION: Tune function
# ============================================== #
def tune_with_setup(parent_run_id: str = None):
    """Main function to run Ray Tune hyperparameter optimization."""
    # Load data ONCE
    log_section("Loading Data", "📦")
    data_version = WORKFLOW_TAGS.dvc_data_version
    train_ds, val_ds, label2id, id2label, metadata = load_data(
        version=data_version,
        batch_size=32,
    )
    num_classes = len(label2id)
    logger.success(f"✨ Loaded data with {num_classes} classes")

    # Build config - simplified for demo
    config = {
        "learning_rate": tune.loguniform(1e-5, 1e-4),  # Narrower range
        "batch_size": tune.choice([16, 32]),
    }

    # Wrap training function with data
    trainable = tune.with_parameters(
        train_emotion_classifier,
        parent_run_id=parent_run_id,
        train_ds=train_ds,
        val_ds=val_ds,
        num_classes=num_classes,
        num_epochs=1,  # Just 1 epoch for demo
    )

    # Wrap with resources - allocate GPU + more CPUs for faster training
    # This will run trials sequentially but each trial will be MUCH faster
    trainable_with_resources = tune.with_resources(
        trainable,
        resources={"cpu": 4, "gpu": 0.5},  # 4 CPUs + 0.5 GPU per trial
    )

    # Configure scheduler
    scheduler = ASHAScheduler(
        metric="val_acc",
        mode="max",
        max_t=1,  # Match num_epochs
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        trainable_with_resources,  # Use the resource-wrapped trainable
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=2,  # Just 2 trials for demo
        ),
        param_space=config,
        run_config=tune.RunConfig(
            name=TRAINING_CONFIG.mlflow_experiment_name,
            stop={"training_iteration": 100},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_score_attribute="val_acc",
                num_to_keep=2,
            ),
        ),
    )

    # Run tuning
    log_section("Starting Ray Tune", "🔍")
    results = tuner.fit()

    return results, train_ds, val_ds, label2id, id2label


# ============================================== #
# 🔹 SECTION: Log Best Model
# ============================================== #
def log_best_model(
    best_result,
    train_ds,
    label2id,
    id2label,
    num_classes: int,
    parent_run_id: str,
):
    """Load best model checkpoint and log to MLflow with input example."""
    log_section("Logging Best Model", "💾")

    # Get best config
    best_config = best_result.config
    logger.info(
        f"Best config: LR={best_config['learning_rate']:.2e}, BS={best_config['batch_size']}"
    )

    # Recreate best model
    best_model = DistilBERTClassificationModule(
        model_name=TRAINING_CONFIG.model_name,
        num_classes=num_classes,
        learning_rate=best_config["learning_rate"],
    )

    # Get single sample WITHOUT materializing whole dataset
    # take_batch returns a dict with tensors directly
    sample_batch = train_ds.take_batch(1)  # Just 1 row

    input_example = {
        "input_ids": sample_batch["input_ids"][:1],
        "attention_mask": sample_batch["attention_mask"][:1],
    }

    # Log to MLflow
    with mlflow.start_run(run_id=parent_run_id):
        signature = mlflow.models.infer_signature(
            model_input=input_example,
            model_output={"predictions": torch.tensor([[0.1] * num_classes])},
        )

        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            artifact_path="best_model",
            signature=signature,
            input_example=input_example,
            registered_model_name=TRAINING_CONFIG.mlflow_registered_model_name,
        )

        # Log label mappings as artifact
        mlflow.log_dict(
            {"label2id": label2id, "id2label": id2label}, "label_mappings.json"
        )

        # Log best metrics
        mlflow.log_metrics(
            {
                "best_val_acc": best_result.metrics.get("val_acc", 0),
                "best_val_loss": best_result.metrics.get("val_loss", 0),
                "best_val_f1": best_result.metrics.get("val_f1", 0),
            }
        )

    logger.success("✅ Best model logged to MLflow")


# ============================================== #
# 🔹 SECTION: Main Entry Point
# ============================================== #
def main():
    """Main entry point - EXACTLY like Ray example."""
    log_section("DistilBERT Emotion Classification - Ray Tune", "🎯")

    # Start MLflow experiment
    with mlflow.start_run(
        run_name="hyperparameter_optimization",
        tags=WORKFLOW_TAGS.model_dump(),
    ) as parent_run:
        parent_run_id = parent_run.info.run_id
        logger.info(f"Started parent MLflow run with ID: {parent_run_id}")

        # Run tuning
        results, train_ds, val_ds, label2id, id2label = tune_with_setup(
            parent_run_id=parent_run_id
        )

        if not results:
            logger.error("No results returned from tuning, run failed.")
            return

        # Get best result
        log_section("Results", "📊")
        best_result = results.get_best_result(metric="val_acc", mode="max")
        logger.success("Best hyperparameters found:")
        logger.info(f"  Learning rate: {best_result.config['learning_rate']:.2e}")
        logger.info(f"  Batch size: {best_result.config['batch_size']}")
        logger.success(f"  Best val_acc: {best_result.metrics.get('val_acc', 0):.4f}")

        # Log best model to MLflow
        log_best_model(
            best_result=best_result,
            train_ds=train_ds,
            label2id=label2id,
            id2label={
                int(k): v for k, v in id2label.items()
            },  # Ensure JSON serializable
            num_classes=len(label2id),
            parent_run_id=parent_run_id,
        )

    logger.success("🎉 Hyperparameter optimization complete!")


if __name__ == "__main__":
    main()
