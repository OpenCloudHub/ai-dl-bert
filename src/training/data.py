"""Data loading and preprocessing for emotion classification."""

import json
import os
from typing import Dict, List, Tuple

import dvc.api
import ray
import s3fs
from pyarrow.fs import FSSpecHandler, PyFileSystem
from ray.data import Dataset
from transformers import AutoTokenizer

from src._utils.logging import get_logger, log_section
from src.training.config import TRAINING_CONFIG

logger = get_logger(__name__)


def get_label_mapping(version: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Fetch label mapping from DVC metadata.

    Args:
        version: DVC version tag (e.g., 'v0.0.1')

    Returns:
        Tuple of (label2id, id2label) mappings
    """
    metadata_content = dvc.api.read(
        TRAINING_CONFIG.dvc_metrics_path,
        repo=TRAINING_CONFIG.dvc_repo,
        rev=version,
    )
    metadata = json.loads(metadata_content)
    classes = metadata["schema"]["features"]["label"]["classes"]

    label2id = {label: idx for idx, label in enumerate(sorted(classes))}
    id2label = {idx: label for label, idx in label2id.items()}

    return label2id, id2label


def tokenize_batch(
    batch: Dict[str, List], tokenizer: AutoTokenizer, label2id: Dict[str, int]
):
    """Tokenize a batch of text data.

    Args:
        batch: Dictionary with 'text' and 'label' keys (numpy arrays when batch_format="numpy")
        tokenizer: HuggingFace tokenizer
        label2id: Mapping from label strings to integers

    Returns:
        Dictionary with tokenized inputs
    """
    # Convert numpy array to list
    texts = batch["text"].tolist()
    labels_raw = batch["label"].tolist()

    # Tokenize text
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=TRAINING_CONFIG.max_length,
        return_tensors="np",
    )

    # Convert labels to integers
    labels = [label2id[label] for label in labels_raw]

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "label": labels,
    }


def load_data(
    version: str,
    batch_size: int = 32,
) -> Tuple[Dataset, Dataset, Dict[str, int], Dict[int, str], dict]:
    """Load and transform datasets from DVC.

    Args:
        version: DVC version tag (e.g., 'v0.0.1')
        batch_size: Batch size for tokenization

    Returns:
        Tuple of (train_ds, val_ds, label2id, id2label, metadata)
    """
    log_section(f"Loading Data Version {version}", "📦")
    logger.info(f"DVC repo: [cyan]{TRAINING_CONFIG.dvc_repo}[/cyan]")

    # Get URLs from DVC
    train_path = dvc.api.get_url(
        TRAINING_CONFIG.dvc_train_data_path,
        repo=TRAINING_CONFIG.dvc_repo,
        rev=version,
    )
    val_path = dvc.api.get_url(
        TRAINING_CONFIG.dvc_val_data_path,
        repo=TRAINING_CONFIG.dvc_repo,
        rev=version,
    )

    # Load metadata
    metadata_content = dvc.api.read(
        TRAINING_CONFIG.dvc_metrics_path,
        repo=TRAINING_CONFIG.dvc_repo,
        rev=version,
    )
    metadata = json.loads(metadata_content)

    logger.info(
        f"Loaded dataset: [bold]{metadata['dataset']['name']}[/bold] [green]({version})[/green]"
    )

    # Get label mappings
    label2id, id2label = get_label_mapping(version)
    logger.info(f"Number of classes: [yellow]{len(label2id)}[/yellow]")
    logger.info(f"Classes: {list(label2id.keys())}")

    # Configure S3 filesystem
    s3_client = s3fs.S3FileSystem(
        anon=False,
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        client_kwargs={
            "verify": False,  # Disable SSL verification for self-signed certs
        },
    )

    # Wrap with PyArrow filesystem
    pa_fs = PyFileSystem(FSSpecHandler(s3_client))

    # Load datasets with custom filesystem
    train_ds = ray.data.read_parquet(train_path, filesystem=pa_fs)
    val_ds = ray.data.read_parquet(val_path, filesystem=pa_fs)

    # Initialize tokenizer
    logger.info(f"Loading tokenizer: [cyan]{'distilbert-base-uncased'}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Apply tokenization
    log_section("Tokenizing Data", "🔤")
    train_ds = train_ds.map_batches(
        lambda batch: tokenize_batch(batch, tokenizer, label2id),
        # batch_format="numpy",
        batch_size=batch_size,
    )
    val_ds = val_ds.map_batches(
        lambda batch: tokenize_batch(batch, tokenizer, label2id),
        # batch_format="numpy",
        batch_size=batch_size,
    )

    logger.success("✨ Data loaded and tokenized")

    return train_ds, val_ds, label2id, id2label, metadata
