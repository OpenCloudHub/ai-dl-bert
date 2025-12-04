# =============================================================================
# model.py - PyTorch Lightning Module for DistilBERT Classification
# =============================================================================
#
# Purpose:
#   Defines the PyTorch Lightning module that wraps DistilBERT for multi-class
#   emotion classification with proper metric tracking.
#
# Why PyTorch Lightning?
#   - Separates model logic from training boilerplate
#   - Built-in logging, checkpointing, and callback support
#   - Easy integration with Ray Tune via TuneReportCheckpointCallback
#
# Model: DistilBERT Base (66M params) → Classification Head → Softmax
# Metrics: Loss (CE), Accuracy, F1 Score (macro)
#
# Usage:
#   model = DistilBERTClassificationModule("distilbert-base-uncased", 6)
#   trainer = L.Trainer(max_epochs=3)
#   trainer.fit(model, train_loader, val_loader)
#
# =============================================================================
"""
PyTorch Lightning module for DistilBERT emotion classification.

Provides a LightningModule with:
- DistilBERT backbone with classification head
- Multi-class accuracy and F1 metrics
- AdamW optimizer with configurable weight decay
"""

import lightning as L
import torch
import torchmetrics
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)
from transformers import DistilBertForSequenceClassification


class DistilBERTClassificationModule(L.LightningModule):
    """
    PyTorch Lightning Module for DistilBERT-based emotion classification.

    This module wraps HuggingFace's DistilBertForSequenceClassification,
    adding proper metric tracking and optimizer configuration for fine-tuning
    on emotion classification tasks.

    Attributes:
        model: DistilBertForSequenceClassification instance
        train_metrics: MetricCollection for training (accuracy, F1)
        val_metrics: MetricCollection for validation (accuracy, F1)
        train_loss: MeanMetric for epoch-averaged training loss
        val_loss: MeanMetric for epoch-averaged validation loss
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
    ):
        """
        Initialize the classification module.

        Args:
            model_name: HuggingFace model identifier (e.g., "distilbert-base-uncased")
            num_classes: Number of emotion classes to predict
            learning_rate: Learning rate for AdamW optimizer
            weight_decay: L2 regularization weight for AdamW
        """
        super().__init__()
        # Save hyperparameters for checkpointing and logging
        self.save_hyperparameters()

        # Load pre-trained DistilBERT with classification head
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

        # Setup metrics using torchmetrics for proper distributed accumulation
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "f1": MulticlassF1Score(num_classes=num_classes),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")

        # Loss trackers for epoch averaging
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through DistilBERT.

        Args:
            input_ids: Tokenized input tensor [batch_size, seq_length]
            attention_mask: Attention mask tensor [batch_size, seq_length]
            labels: Optional ground truth labels for loss computation

        Returns:
            SequenceClassifierOutput with logits and optional loss
        """
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        """Execute single training step and log metrics."""
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["label"])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        # Update metrics
        self.train_loss(loss)
        self.train_metrics(preds, batch["label"])

        # Log metrics (on_epoch=True for epoch-averaged values)
        self.log(
            "train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Execute single validation step and log metrics."""
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["label"])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        # Update metrics
        self.val_loss(loss)
        self.val_metrics(preds, batch["label"])

        # Log metrics
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """
        Configure AdamW optimizer for fine-tuning.

        AdamW is preferred for transformer fine-tuning as it properly
        decouples weight decay from gradient updates.
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
