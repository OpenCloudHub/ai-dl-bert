# serving/bert_predictor.py
from typing import Dict, List

import mlflow
import torch


class BERTPredictor:
    def __init__(self, model_uri: str):
        """Load model from MLflow."""
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.eval()

    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run inference."""
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        return outputs

    def predict_batch(self, texts: List[str], tokenizer) -> List[Dict]:
        """Predict from raw text."""
        # Tokenize
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        # Predict
        logits = self.predict(inputs["input_ids"], inputs["attention_mask"])

        # Convert to predictions
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)

        return [
            {"prediction": pred.item(), "probabilities": probs.tolist()}
            for pred, probs in zip(predictions, probabilities)
        ]


# Usage in FastAPI/Ray
predictor = BERTPredictor("models:/bert-emotion-classifier/production")
