# =============================================================================
# tune.py - Ray Tune Hyperparameter Search with MLflow Integration
# =============================================================================
#
# Purpose:
#   Orchestrates distributed hyperparameter optimization using Ray Tune while
#   maintaining a unified experiment tracking hierarchy in MLflow.
#
# Key Concepts:
#   - Parent-Child MLflow Runs: Parent run for entire search, child runs per trial
#   - Ray Tune + Lightning: TuneReportCheckpointCallback for metrics & checkpoints
#   - Automatic Model Registration: Best model registered to MLflow Model Registry
#
# Usage:
#   # Local development
#   python src/training/tune.py --num-epochs 3 --num-samples 4 --limit 1000
#
#   # As Ray job
#   RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . \
#     -- python src/training/tune.py --num-epochs 5 --num-samples 8
#
# Required Environment Variables:
#   MLFLOW_TRACKING_URI, DVC_DATA_VERSION
#
# =============================================================================
"""
Ray Tune + MLflow + PyTorch Lightning hyperparameter optimization.

Provides distributed hyperparameter search with parent-child MLflow run
hierarchy and automatic best model registration.
"""

import os
import tempfile

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse

import lightning as L
import mlflow
import numpy as np
import ray
import ray.tune
import torch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from ray import tune
from ray.train import Checkpoint
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

from src._utils.logging import get_logger, log_section
from src.training.config import TRAINING_CONFIG, WORKFLOW_TAGS
from src.training.data import load_and_prepare_data
from src.training.model import DistilBERTClassificationModule

logger = get_logger(__name__)


# =============================================================================
# Custom Callback to Track MLflow Run ID in Checkpoints
# =============================================================================
class TuneReportCheckpointWithMLflow(TuneReportCheckpointCallback):
    """
    Extended TuneReportCheckpointCallback that saves the MLflow run ID
    alongside the model checkpoint, enabling the main process to identify
    which MLflow run corresponds to the best trial.

    Also logs metrics to MLflow at each validation end to ensure metrics
    are captured even when ASHA early-stops a trial.
    """

    def __init__(self, mlflow_run_id: str, **kwargs):
        super().__init__(**kwargs)
        self.mlflow_run_id = mlflow_run_id

    def _handle(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Override to include MLflow run ID in checkpoint and log metrics."""
        if trainer.sanity_checking:
            return

        # Get metrics
        metrics = {}
        for key, value in self._metrics.items():
            if value in trainer.callback_metrics:
                metrics[key] = trainer.callback_metrics[value].item()

        # Log metrics to MLflow immediately (important for early-stopped trials)
        # This ensures metrics are captured before ASHA potentially kills the process
        try:
            with mlflow.start_run(run_id=self.mlflow_run_id, nested=True):
                epoch = trainer.current_epoch + 1
                # Log validation metrics (same as what Ray Tune sees)
                mlflow.log_metrics(
                    {
                        "val_loss": metrics.get("loss", 0),
                        "val_acc": metrics.get("accuracy", 0),
                        "val_f1": metrics.get("f1", 0),
                        "epoch": epoch,
                    },
                    step=epoch,
                )
                # Also log training metrics if available
                train_metrics = {
                    k: v.item() if hasattr(v, "item") else v
                    for k, v in trainer.callback_metrics.items()
                    if k.startswith("train_")
                }
                if train_metrics:
                    mlflow.log_metrics(train_metrics, step=epoch)
        except Exception:
            pass  # Don't fail training if MLflow logging fails

        # Create checkpoint with MLflow run ID
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model checkpoint
            ckpt_path = os.path.join(tmpdir, "checkpoint")
            trainer.save_checkpoint(ckpt_path)

            # Save MLflow run ID
            run_id_path = os.path.join(tmpdir, "mlflow_run_id.txt")
            with open(run_id_path, "w") as f:
                f.write(self.mlflow_run_id)

            # Report to Ray Tune with checkpoint
            checkpoint = Checkpoint.from_directory(tmpdir)
            tune.report(metrics, checkpoint=checkpoint)


# =============================================================================
# Training Function (Executed per Trial)
# =============================================================================
def train_model(config, train_dataset, val_dataset, num_classes, parent_run_id):
    """
    Training function executed for each Ray Tune trial.

    This function is called by Ray Tune for each hyperparameter combination.
    It creates a nested MLflow child run, trains the model using PyTorch
    Lightning, and reports metrics back to Ray Tune for hyperparameter
    selection.

    Args:
        config: Dict containing hyperparameters for this trial:
            - learning_rate: Float, learning rate for AdamW optimizer
            - batch_size: Int, batch size for training
            - num_epochs: Int, number of training epochs
        train_dataset: PyTorch Dataset for training
        val_dataset: PyTorch Dataset for validation
        num_classes: Number of emotion classes
        parent_run_id: MLflow run ID of the parent hyperparameter search run

    Note:
        The function saves the MLflow run ID to the Ray checkpoint directory,
        allowing the main process to identify which MLflow run corresponds
        to the best trial for model registration.
    """
    # Get Ray Tune trial info for consistent naming
    trial_id = ray.tune.get_context().get_trial_id()
    trial_name = ray.tune.get_context().get_trial_name()

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
    # Ensure MLflow connection is established before starting run
    mlflow.set_tracking_uri(TRAINING_CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(TRAINING_CONFIG.mlflow_experiment_name)

    # Verify experiment exists (forces connection establishment)
    experiment = mlflow.get_experiment_by_name(TRAINING_CONFIG.mlflow_experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{TRAINING_CONFIG.mlflow_experiment_name}' not found"
        )

    # Start the run with the same name as Ray Tune trial for easy correlation
    run = mlflow.start_run(
        run_name=trial_name,  # Use Ray Tune's trial name for consistency
        nested=True,
        tags=WORKFLOW_TAGS.model_dump(),
    )
    mlflow_run_id = run.info.run_id

    try:
        # Set parent tag manually (more reliable in distributed setting)
        mlflow.MlflowClient().set_tag(
            mlflow_run_id, "mlflow.parentRunId", parent_run_id
        )
        # Also tag with Ray Tune trial ID for traceability
        mlflow.set_tag("ray_tune_trial_id", trial_id)
        mlflow.set_tag("ray_tune_trial_name", trial_name)

        # Log all hyperparameters from the search space
        mlflow.log_params(
            {
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "num_epochs": config["num_epochs"],
                "model_name": TRAINING_CONFIG.model_name,
            }
        )

        # Custom callback that saves MLflow run ID with checkpoint
        tune_callback = TuneReportCheckpointWithMLflow(
            mlflow_run_id=mlflow_run_id,
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

        # NOTE: We intentionally do NOT use mlflow.pytorch.autolog() here.
        # Our TuneReportCheckpointWithMLflow callback handles metric logging
        # to ensure metrics exactly match what Ray Tune reports.

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Log final training state
        mlflow.log_metric(
            "completed_all_epochs",
            int(trainer.current_epoch + 1 == config["num_epochs"]),
        )

    finally:
        # Always end the MLflow run, even if ASHA terminates us early
        mlflow.end_run()


# =============================================================================
# Main Orchestration
# =============================================================================
def main():
    """
    Main orchestration function for hyperparameter search.

    Workflow:
        1. Load and prepare data from DVC registry
        2. Create MLflow parent run for the search
        3. Configure Ray Tune with search space and resources
        4. Execute distributed trials
        5. Identify best trial and register model to MLflow

    The function handles the complete lifecycle of a hyperparameter search,
    from data loading to model registration, with proper cleanup of Ray
    resources at the end.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=2)  # Number of HP trials
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

        # Log parent run params (including ASHA scheduler config)
        mlflow.log_params(
            {
                "num_epochs": args.num_epochs,
                "data_limit": args.limit,
                "scheduler": "ASHAScheduler",
                "asha_metric": "accuracy",
                "asha_mode": "max",
                "asha_grace_period": 2,
                "asha_reduction_factor": 2,
                "asha_max_t": args.num_epochs,
                "asha_brackets": 1,
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

        # Search space - wider LR range for better exploration
        search_space = {
            "learning_rate": tune.loguniform(1e-5, 1e-4),  # Wider range
            "batch_size": tune.choice([16, 32]),
            "num_epochs": args.num_epochs,
        }

        gpu_available = torch.cuda.is_available()
        resources = {"cpu": 4, "gpu": 1 if gpu_available else 0}
        logger.info(f"Resources per trial: [cyan]{resources}[/cyan]")

        # ASHA Scheduler for early stopping of underperforming trials
        # - grace_period: Minimum epochs before a trial can be stopped
        # - max_t: Maximum epochs a trial can run
        # - reduction_factor: Fraction of trials to keep each round (1/factor survive)
        # - brackets: Number of brackets (1 recommended by ASHA authors)
        asha_scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="accuracy",
            mode="max",
            max_t=args.num_epochs,
            grace_period=2,  # Minimum 2 epochs - ensures meaningful validation metrics
            reduction_factor=2,  # Aggressive: halve trials each round
            brackets=1,
        )

        tuner = tune.Tuner(
            tune.with_resources(trainable, resources=resources),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                scheduler=asha_scheduler,
                num_samples=args.num_samples,
            ),
        )

        log_section("Running Hyperparameter Trials", "🔍")
        logger.info("This may take several minutes...")
        results = tuner.fit()

        # =================================================================
        # Results Analysis
        # =================================================================
        if results.errors:
            logger.error(f"Errors in Trials: {results.errors}")
        else:
            logger.success("✨ No errors in trials")

        # =================================================================
        # End Orphaned MLflow Runs
        # =================================================================
        # When ASHA terminates trials early, the MLflow runs may not be
        # properly closed because Ray kills the process. We need to
        # explicitly end all child runs that are still in 'RUNNING' state.
        client = mlflow.MlflowClient()
        child_runs = client.search_runs(
            experiment_ids=[
                mlflow.get_experiment_by_name(
                    TRAINING_CONFIG.mlflow_experiment_name
                ).experiment_id
            ],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
        )
        orphaned_count = 0
        for child_run in child_runs:
            if child_run.info.status == "RUNNING":
                client.set_terminated(child_run.info.run_id, status="FINISHED")
                orphaned_count += 1
        if orphaned_count > 0:
            logger.info(
                f"Closed [yellow]{orphaned_count}[/yellow] orphaned MLflow runs (early-stopped by ASHA)"
            )

        # =================================================================
        # ASHA Early Stopping Analysis
        # =================================================================
        log_section("ASHA Early Stopping Summary", "⏱️")
        all_results = results.get_dataframe()
        early_stopped = len(
            all_results[all_results["training_iteration"] < args.num_epochs]
        )
        completed_full = len(
            all_results[all_results["training_iteration"] == args.num_epochs]
        )
        total_epochs_saved = sum(
            args.num_epochs - r for r in all_results["training_iteration"]
        )

        logger.info(
            f"Trials stopped early: [yellow]{early_stopped}[/yellow] / {len(all_results)}"
        )
        logger.info(f"Trials completed full training: [green]{completed_full}[/green]")
        logger.info(
            f"Total epochs saved by early stopping: [cyan]{total_epochs_saved}[/cyan]"
        )

        # Log early stopping stats to MLflow
        mlflow.log_metrics(
            {
                "asha_trials_stopped_early": early_stopped,
                "asha_trials_completed": completed_full,
                "asha_epochs_saved": total_epochs_saved,
            }
        )

        # Get best result
        log_section("Best Result (by val_acc)", "🏆")
        best_result = results.get_best_result(metric="accuracy", mode="max")
        # Extract trial name from path
        # Path format: .../train_model_XXXXX_NNNNN_N_batch_size=..._2025-.../
        # We want: train_model_XXXXX_NNNNN
        path_basename = os.path.basename(best_result.path)
        # Split by underscore and take first 4 parts: train_model_XXXXX_NNNNN
        parts = path_basename.split("_")
        if len(parts) >= 4 and parts[0] == "train" and parts[1] == "model":
            best_trial_name = "_".join(parts[:4])  # train_model_XXXXX_NNNNN
        else:
            best_trial_name = path_basename
        logger.info("[dim]Selected trial with highest validation accuracy[/dim]")
        logger.success(f"Trial: [yellow]{best_trial_name}[/yellow]")
        logger.success(
            f"Learning rate: [green]{best_result.config['learning_rate']:.2e}[/green]"
        )
        logger.success(f"Batch size: [green]{best_result.config['batch_size']}[/green]")
        logger.success(
            f"Val accuracy: [green]{best_result.metrics.get('accuracy', 0):.4f}[/green]"
        )
        logger.success(f"Val loss: [green]{best_result.metrics['loss']:.4f}[/green]")
        logger.success(f"Val F1: [green]{best_result.metrics.get('f1', 0):.4f}[/green]")

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

                    # Move model to CPU for logging
                    model = model.cpu()

                    # Create model signature for transformer model
                    # Input: input_ids and attention_mask tensors of shape (batch, seq_len)
                    # Output: logits tensor of shape (batch, num_classes)
                    input_schema = Schema(
                        [
                            TensorSpec(
                                np.dtype(np.int64),
                                (-1, TRAINING_CONFIG.max_length),
                                "input_ids",
                            ),
                            TensorSpec(
                                np.dtype(np.int64),
                                (-1, TRAINING_CONFIG.max_length),
                                "attention_mask",
                            ),
                        ]
                    )
                    output_schema = Schema(
                        [
                            TensorSpec(
                                np.dtype(np.float32), (-1, num_classes), "logits"
                            ),
                        ]
                    )
                    signature = ModelSignature(
                        inputs=input_schema, outputs=output_schema
                    )

                    logger.info(
                        f"Registering model as: [cyan]{TRAINING_CONFIG.mlflow_registered_model_name}[/cyan]"
                    )
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        name="model",
                        registered_model_name=TRAINING_CONFIG.mlflow_registered_model_name,
                        signature=signature,
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
