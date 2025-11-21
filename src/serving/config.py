from pydantic_settings import BaseSettings


class ServingConfig(BaseSettings):
    model_name: str = "distilbert-base-uncased"
    api_name: str = "emotion-classifier"
    request_max_length: int = 128


SERVING_CONFIG = ServingConfig()
