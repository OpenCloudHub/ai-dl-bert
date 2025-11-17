# ai-dl-bert
Deep learning NLP workflows with BERT transformers demonstrating advanced MLOps and DVC integration


https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/text-transformers.html



mlflow server --host 0.0.0.0 --port 8081

mlflow server \
  --host 0.0.0.0 \
  --port 8082 \
  --backend-store-uri file:./output/mlruns

export MLFLOW_TRACKING_URI=http://172.17.0.1:8081
export MLFLOW_EXPERIMENT_NAME=emotion-classification-bert
export MLFLOW_REGISTERED_MODEL_NAME=emotion-classifier


uv run src/train.py

RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python src/training/tune.py



## Issues

Current Issues when using mlflow logger and log_model: True
- https://github.com/Lightning-AI/pytorch-lightning/issues/20932



[project]
name = "ai-dl-bert"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "datasets>=4.0.0",
    "hydra-core>=1.3.2",
    "lightning==2.5.5",
    "mlflow>=3.6.0",
    "ray[data,serve,train,tune]>=2.51.1",
    "rich==14.0.0",
    "seaborn>=0.13.2",
    "transformers>=4.55.0",
]
