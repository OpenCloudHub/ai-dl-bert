"""
Simple Ray Tune + MLflow + Pytorch Lightning integration.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse

import lightning as L
import mlflow
import ray
import ray.tune
import torch
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from torch.utils.data import DataLoader

from src._utils.logging import get_logger, log_section
from src.training.config import TRAINING_CONFIG, WORKFLOW_TAGS
from src.training.data import load_and_prepare_data
from src.training.model import DistilBERTClassificationModule

logger = get_logger(__name__)


# ============================================== #
# Training Function
# ============================================== #
def train_model(config, train_dataset, val_dataset, num_classes, parent_run_id):
    """Training function for each trial."""

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    # Create model
    model = DistilBERTClassificationModule(
        model_name=TRAINING_CONFIG.model_name,
        num_classes=num_classes,
        learning_rate=config["learning_rate"],
    )

    # Setup MLflow child run
    mlflow.set_tracking_uri(TRAINING_CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(TRAINING_CONFIG.mlflow_experiment_name)

    with mlflow.start_run(
        run_name=f"trial_lr{config['learning_rate']:.2e}",
        nested=True,
        tags=WORKFLOW_TAGS.model_dump(),
    ) as run:
        # Set parent manually (more reliable in distributed setting)
        mlflow.MlflowClient().set_tag(
            run.info.run_id, "mlflow.parentRunId", parent_run_id
        )

        # Log hyperparameters
        mlflow.log_params(
            {
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "num_epochs": config["num_epochs"],
            }
        )

        # Callback for Ray Tune
        tune_callback = TuneReportCheckpointCallback(
            metrics={"loss": "val_loss", "accuracy": "val_acc", "f1": "val_f1"},
            on="validation_end",
        )

        # Trainer
        trainer = L.Trainer(
            max_epochs=config["num_epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[tune_callback],
            enable_progress_bar=True,
            enable_checkpointing=False,
            logger=False,
        )

        # Enable MLflow autolog
        mlflow.pytorch.autolog(
            log_models=False,
            log_datasets=False,
            log_model_signatures=False,
            checkpoint=False,  # Ray Tune handles checkpoints
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Save run id to checkpoint
        checkpoint = tune.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                run_id_file = os.path.join(checkpoint_dir, "mlflow_run_id.txt")
                with open(run_id_file, "w") as f:
                    f.write(run.info.run_id)


# ============================================== #
# Main Orchestration
# ============================================== #
def main():
    """Main function."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=2)
    args = parser.parse_args()

    log_section("Ray Tune + MLflow Hyperparameter Search", "🎯")
    logger.info(f"DVC Version: [cyan]{WORKFLOW_TAGS.dvc_data_version}[/cyan]")
    logger.info(f"Docker Tag: [cyan]{WORKFLOW_TAGS.docker_image_tag}[/cyan]")
    logger.info(f"Workflow UID: [cyan]{WORKFLOW_TAGS.argo_workflow_uid}[/cyan]")

    # Load data
    train_dataset, val_dataset, label2id, id2label, num_classes = load_and_prepare_data(
        limit=args.limit
    )

    # Setup MLflow parent run
    log_section("Starting MLflow Experiment", "📊")
    mlflow.set_tracking_uri(TRAINING_CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(TRAINING_CONFIG.mlflow_experiment_name)

    with mlflow.start_run(
        run_name="hyperparameter_search", tags=WORKFLOW_TAGS.model_dump()
    ) as parent_run:
        logger.info(f"Parent run ID: [yellow]{parent_run.info.run_id}[/yellow]")

        # Log parent run params
        mlflow.log_params(
            {
                "num_trials": 2,
                "num_epochs": args.num_epochs,
                "data_limit": args.limit,
            }
        )

        # Run trials with tune.with_parameters to pass data
        trainable = tune.with_parameters(
            train_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_classes=num_classes,
            parent_run_id=parent_run.info.run_id,
        )

        search_space = {
            "learning_rate": tune.loguniform(1e-5, 5e-5),
            "batch_size": tune.choice([16, 32]),
            "num_epochs": args.num_epochs,
        }

        gpu_available = torch.cuda.is_available()
        resources = {"cpu": 4, "gpu": 1 if gpu_available else 0}
        logger.info(f"Resources per trial: [cyan]{resources}[/cyan]")

        tuner = tune.Tuner(
            tune.with_resources(trainable, resources=resources),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                num_samples=4,
            ),
        )

        log_section("Running Hyperparameter Trials", "🔍")
        logger.info("This may take several minutes...")
        results = tuner.fit()

        # Get errors
        if results.errors:
            logger.error(f"Errors in Trials: {results.errors}")
        else:
            logger.success("✨ No errors in trials")

        # Get best result
        log_section("Best Result", "🏆")
        best_result = results.get_best_result(metric="loss", mode="min")
        logger.success(
            f"Best learning rate: [green]{best_result.config['learning_rate']:.2e}[/green]"
        )
        logger.success(
            f"Best batch size: [green]{best_result.config['batch_size']}[/green]"
        )
        logger.success(
            f"Best val loss: [green]{best_result.metrics['loss']:.4f}[/green]"
        )
        logger.success(
            f"Best val accuracy: [green]{best_result.metrics.get('accuracy', 0):.4f}[/green]"
        )

        # Log summary to parent
        mlflow.log_metrics(
            {
                "best_val_loss": best_result.metrics["loss"],
                "best_val_acc": best_result.metrics.get("accuracy", 0),
                "best_val_f1": best_result.metrics.get("f1", 0),
            }
        )
        mlflow.log_params(
            {
                "best_learning_rate": best_result.config["learning_rate"],
                "best_batch_size": best_result.config["batch_size"],
            }
        )

    # Log best model
    log_section("Logging Best Model", "💾")
    if best_result.checkpoint:
        # Get the best MLflow run ID from the checkpoint
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            try:
                best_mlflow_run_id_file = os.path.join(
                    checkpoint_dir, "mlflow_run_id.txt"
                )
                with open(best_mlflow_run_id_file, "r") as f:
                    best_mlflow_run_id = f.read().strip()
                logger.info(
                    f"Best trial MLflow run ID: [yellow]{best_mlflow_run_id}[/yellow]"
                )
            except FileNotFoundError:
                logger.warning("⚠️  No MLflow run ID file found in the best checkpoint")
                best_mlflow_run_id = None

        if best_mlflow_run_id:
            with mlflow.start_run(run_id=best_mlflow_run_id):
                with best_result.checkpoint.as_directory() as checkpoint_dir:
                    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

                    logger.info("Loading best model from checkpoint...")
                    model = DistilBERTClassificationModule.load_from_checkpoint(
                        checkpoint_path,
                        model_name=TRAINING_CONFIG.model_name,
                        num_classes=num_classes,
                        learning_rate=best_result.config["learning_rate"],
                    )

                    logger.info(
                        f"Registering model as: [cyan]{TRAINING_CONFIG.mlflow_registered_model_name}[/cyan]"
                    )
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="model",
                        registered_model_name=TRAINING_CONFIG.mlflow_registered_model_name,
                    )

                    mlflow.log_dict(
                        {"label2id": label2id, "id2label": id2label}, "labels.json"
                    )

                    mlflow.set_tag("model_selection", "best_from_tune")

                    logger.success(
                        f"✨ Model logged to run: [yellow]{best_mlflow_run_id}[/yellow]"
                    )
    else:
        logger.warning("⚠️  No checkpoint found for best result")

    ray.shutdown()
    log_section("Complete", "🎉")
    logger.success("Hyperparameter search finished successfully!")


if __name__ == "__main__":
    main()
