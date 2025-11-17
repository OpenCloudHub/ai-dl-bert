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
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        learning_rate: float = 2e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )

        # Metrics
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "f1": MulticlassF1Score(num_classes=num_classes),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["label"])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        self.train_loss(loss)
        self.train_metrics(preds, batch["label"])

        self.log(
            "train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["label"])
        loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=1)

        self.val_loss(loss)
        self.val_metrics(preds, batch["label"])

        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01
        )
