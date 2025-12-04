# =============================================================================
# config.py - Serving Configuration
# =============================================================================
#
# Purpose:
#   Configuration settings for the Ray Serve inference API using Pydantic
#   Settings. Provides consistent tokenization and API settings that must
#   match the training configuration.
#
# Important:
#   The model_name and request_max_length MUST match the values used during
#   training to ensure correct tokenization. Mismatched settings will cause
#   prediction errors.
#
# Usage:
#   from src.serving.config import SERVING_CONFIG
#   tokenizer = AutoTokenizer.from_pretrained(SERVING_CONFIG.model_name)
#
# =============================================================================
"""
Serving configuration using Pydantic Settings.

Provides consistent settings for inference that match training configuration.
"""

from pydantic_settings import BaseSettings


class ServingConfig(BaseSettings):
    """
    Configuration for the serving API.

    Attributes:
        model_name: HuggingFace model identifier for tokenizer
                   (must match training config)
        api_name: Name of the API service for metadata
        request_max_length: Maximum sequence length for tokenization
                           (must match training config)
    """

    model_name: str = "distilbert-base-uncased"  # Must match training
    api_name: str = "emotion-classifier"
    request_max_length: int = 128  # Must match training max_length


# Singleton instance for import
SERVING_CONFIG = ServingConfig()
