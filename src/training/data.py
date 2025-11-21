import json

import dvc.api
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src._utils.logging import get_logger, log_section
from src.training.config import TRAINING_CONFIG, WORKFLOW_TAGS

logger = get_logger(__name__)


class EmotionDataset(Dataset):
    """Dataset that returns dict format the model expects."""

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


def load_and_prepare_data(limit=None):
    """Load data from DVC once and prepare datasets."""
    log_section("Loading Data from DVC", "📦")

    logger.info(f"DVC version: [cyan]{WORKFLOW_TAGS.dvc_data_version}[/cyan]")
    logger.info(f"DVC repo: [cyan]{TRAINING_CONFIG.dvc_repo}[/cyan]")

    # Load metadata
    metadata_content = dvc.api.read(
        TRAINING_CONFIG.dvc_metrics_path,
        repo=TRAINING_CONFIG.dvc_repo,
        rev=WORKFLOW_TAGS.dvc_data_version,
    )
    metadata = json.loads(metadata_content)
    classes = metadata["schema"]["features"]["label"]["classes"]

    label2id = {label: idx for idx, label in enumerate(sorted(classes))}
    id2label = {idx: label for label, idx in label2id.items()}

    logger.info(f"Number of classes: [yellow]{len(label2id)}[/yellow]")

    # Load data
    with dvc.api.open(
        TRAINING_CONFIG.dvc_train_data_path,
        repo=TRAINING_CONFIG.dvc_repo,
        rev=WORKFLOW_TAGS.dvc_data_version,
        mode="rb",
    ) as f:
        train_df = pd.read_parquet(f)

    with dvc.api.open(
        TRAINING_CONFIG.dvc_val_data_path,
        repo=TRAINING_CONFIG.dvc_repo,
        rev=WORKFLOW_TAGS.dvc_data_version,
        mode="rb",
    ) as f:
        val_df = pd.read_parquet(f)

    if limit:
        logger.warning(f"⚠️  Limiting datasets to {limit} samples for testing")
        train_df = train_df.head(limit)
        val_df = val_df.head(limit)

    logger.info(f"Train samples: [green]{len(train_df):,}[/green]")
    logger.info(f"Val samples: [green]{len(val_df):,}[/green]")

    # Tokenize
    log_section("Tokenizing Data", "🔤")
    logger.info(f"Loading tokenizer: [cyan]{TRAINING_CONFIG.model_name}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(TRAINING_CONFIG.model_name)

    train_encoded = tokenizer(
        train_df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=TRAINING_CONFIG.max_length,
        return_tensors="pt",
    )

    val_encoded = tokenizer(
        val_df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=TRAINING_CONFIG.max_length,
        return_tensors="pt",
    )

    # Create datasets with proper format
    train_dataset = EmotionDataset(
        train_encoded["input_ids"],
        train_encoded["attention_mask"],
        torch.tensor([label2id[l] for l in train_df["label"]], dtype=torch.long),
    )

    val_dataset = EmotionDataset(
        val_encoded["input_ids"],
        val_encoded["attention_mask"],
        torch.tensor([label2id[l] for l in val_df["label"]], dtype=torch.long),
    )

    logger.success("✨ Data loaded and tokenized successfully")
    return train_dataset, val_dataset, label2id, id2label, len(label2id)
