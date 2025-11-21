<a id="readme-top"></a>

<!-- PROJECT LOGO & TITLE -->

<div align="center">
  <a href="https://github.com/opencloudhub">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-light.svg">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-dark.svg">
    <!-- Fallback -->
    <img alt="OpenCloudHub Logo" src="https://raw.githubusercontent.com/opencloudhub/.github/main/assets/brand/assets/logos/primary-logo-dark.svg" style="max-width:700px; max-height:175px;">
  </picture>
  </a>

<h1 align="center">Emotion Classification - MLOps Demo</h1>

<p align="center">
    DistilBERT emotion classification with  MLOps pipeline featuring PyTorch Lightning, MLflow tracking, Ray Tune hyperparameter optimization, and Ray Serve deployment.<br />
    <a href="https://github.com/opencloudhub"><strong>Explore OpenCloudHub »</strong></a>
  </p>
</div>

______________________________________________________________________

<details>
  <summary>📑 Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

______________________________________________________________________

<h2 id="about">🎯 About</h2>

This repository demonstrates a MLOps pipeline for text classification using emotion detection as the example use case. It showcases the integration of modern ML tooling including PyTorch Lightning for structured training, MLflow for experiment tracking and model registry, Ray Tune for distributed hyperparameter optimization, and Ray Serve for scalable model deployment.

The project uses DistilBERT (a distilled version of BERT) fine-tuned on emotion-labeled text data, with data versioning managed through [DVC Registry](https://github.com/OpenCloudHub/data-registry). This serves as a demo implementation for the OpenCloudHub project, demonstrating practices for ML workflows that can scale from local development to production clusters.

**Key Learning Points:**

- Integration of PyTorch Lightning with MLflow for experiment tracking
- Parent-child MLflow runs for hyperparameter search organization
- Ray Tune callbacks for distributed training with automatic checkpointing
- Model registration and versioning in MLflow Model Registry
- Zero-downtime model serving with Ray Serve
- DVC integration for reproducible dataset versioning

______________________________________________________________________

<h2 id="features">✨ Features</h2>

### MLOps Pipeline

- 🔬 **Experiment Tracking**: MLflow integration with parent-child run hierarchy for hyperparameter tuning
- 📊 **Model Registry**: Automatic model registration with versioning and metadata
- 🎯 **Hyperparameter Optimization**: Ray Tune with ASHA scheduler for efficient search
- ⚡ **Distributed Training**: Ray + PyTorch Lightning for scalable model training
- 🚀 **Model Serving**: FastAPI + Ray Serve for production inference with hot model reloading

### Development & Deployment

- 🐳 **Containerized Environment**: Docker-based development with UV package manager
- 📦 **Data Versioning**: DVC integration for reproducible dataset management
- 🧪 **VS Code DevContainer**: Pre-configured development environment
- 🔄 **CI/CD Ready**: GitHub Actions workflows for automated CI/CD and training pipelines

### Key Integrations

#### PyTorch Lightning + MLflow

- Automatic metric logging during training and validation
- Model checkpointing integrated with MLflow artifact storage
- Hyperparameter logging with experiment comparison

#### Ray Tune + MLflow

- Distributed hyperparameter search with parent run tracking
- Child runs for each trial automatically nested in MLflow
- Best model selection and automatic registration to Model Registry

#### Ray Serve + MLflow

- Direct model loading from MLflow Model Registry
- Hot model updates without service restart (using `reconfigure`)
- Automatic model metadata exposure via `/info` endpoint

______________________________________________________________________

<h2 id="architecture">🏗️ Architecture</h2>

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Versioning (DVC)                   │
│  emotion-v0.3.0 → GitHub Data Registry → Local Cache       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Hyperparameter Optimization (Ray Tune)         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Parent MLflow Run: hyperparameter_search           │  │
│  │  ├─ Trial 1 (Child Run): LR=2e-5, BS=16            │  │
│  │  ├─ Trial 2 (Child Run): LR=5e-5, BS=32            │  │
│  │  ├─ Trial 3 (Child Run): LR=3e-5, BS=16            │  │
│  │  └─ Trial 4 (Child Run): LR=4e-5, BS=32            │  │
│  │  → Best model selected and checkpointed             │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Training (PyTorch Lightning)                   │
│  DistilBERT Model                                           │
│  ├─ Automatic metric logging (loss, accuracy, F1)          │
│  ├─ Checkpointing best model                               │
│  └─ Label mappings saved as artifacts                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Registry (MLflow)                        │
│  models:/ci.emotion-classifier/1                            │
│  ├─ Model artifacts (PyTorch Lightning checkpoint)         │
│  ├─ Model signature (input/output schema)                  │
│  ├─ Label mappings (labels.json)                           │
│  └─ Training metadata (DVC version, hyperparameters)       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Serving (Ray Serve + FastAPI)           │
│  GET  /health      → Service health status                 │
│  GET  /info        → Model metadata & emotion labels       │
│  POST /predict     → Batch emotion classification          │
│  └─ Hot reload: Update model without service restart       │
└─────────────────────────────────────────────────────────────┘
```

**Integration Flow:**

1. **Data**: DVC fetches versioned dataset from GitHub registry
1. **Training**: PyTorch Lightning trains model, logs to MLflow
1. **Optimization**: Ray Tune explores hyperparameter space with nested MLflow runs
1. **Registry**: Best model registered to MLflow with metadata
1. **Serving**: Ray Serve loads model from registry for inference

______________________________________________________________________

<h2 id="getting-started">🚀 Getting Started</h2>

### Prerequisites

- Docker
- VS Code with DevContainers extension (recommended)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/opencloudhub/ai-dl-bert.git
   cd ai-dl-bert
   ```

1. **Open in DevContainer** (Recommended)

   VSCode: `Ctrl+Shift+P` → `Dev Containers: Rebuild and Reopen in Container`

   Or **setup locally without DevContainer**:

   ```bash
   # Install UV
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync --dev
   ```

1. **Start local MLflow tracking server**

   ```bash
   mlflow server \
     --host 0.0.0.0 \
     --port 8081 \
     --backend-store-uri file:./output/mlruns \
     --default-artifact-root file:./output/mlruns/artifacts
   ```

   Access MLflow UI at `http://localhost:8081`

1. **Configure environment**

   ```bash
   source .env
   ```

1. **Start local Ray cluster**

   ```bash
   ray start --head --num-cpus 12
   ```

   Access Ray dashboard at `http://127.0.0.1:8265`

You're now ready to train and serve models locally!

______________________________________________________________________

<h2 id="usage">📖 Usage</h2>

### Hyperparameter Tuning

**Submit as Ray job:**

```bash
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit \
  --working-dir . \
  -- python src/training/tune.py --num-epochs 3
```

**What happens:**

- Creates parent MLflow run for the hyperparameter search
- Launches 4 trials with different hyperparameter combinations
- Each trial logs metrics as a nested child run
- Best model automatically registered to MLflow Model Registry
- Label mappings saved as artifacts

**View results:**

- MLflow UI: `http://localhost:8081` - Compare trials and view metrics
- Ray Dashboard: `http://127.0.0.1:8265` - Monitor ray cluster

### Model Serving

**Start the API server:**

```bash
# Development mode with hot reload
serve run src.serving.serve:app_builder \
  model_uri="models:/ci.emotion-classifier/1" \
  --reload

# Config deployment
serve build src.serving.serve:app_builder \
  -o src/serving/serve_config.yaml

serve deploy src/serving/serve_config.yaml
```

**Access the API:**

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Model info: `http://localhost:8000/info`
- Root: `http://localhost:8000/`
- Predict: `http://localhost:8000/predict`

**Example prediction request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I love this product!",
      "This makes me so angry",
      "I am feeling sad today"
    ]
  }'
```

**Response:**

```json
{
  "predictions": [
    {
      "text": "I love this product!",
      "emotion": "joy",
      "confidence": 0.92,
      "all_scores": {
        "joy": 0.92,
        "sadness": 0.02,
        "anger": 0.01,
        "fear": 0.02,
        "love": 0.02,
        "surprise": 0.01
      }
    }
  ],
  "model_uri": "models:/ci.emotion-classifier/1",
  "timestamp": "2025-11-17T22:45:30Z",
  "processing_time_ms": 45.2
}
```

**Hot model updates** (without service restart):

```python
import requests

# Update to a new model version
requests.post(
    "http://localhost:8000/-/routes",
    json={
        "route_config": {
            "user_config": {"model_uri": "models:/ci.emotion-classifier/2"}
        }
    },
)
```

Or use the interactive Swagger UI at `http://localhost:8000/docs`

______________________________________________________________________

<h2 id="project-structure">📁 Project Structure</h2>

```
ai-dl-bert/
├── src/
│   ├── training/                       # Training and optimization
│   │   ├── tune.py                     # Ray Tune + MLflow hyperparameter search
│   │   ├── data.py                     # DVC data loading and preprocessing
│   │   ├── model.py                    # PyTorch Lightning module
│   │   └── config.py                   # Training configuration
│   ├── serving/                        # Model serving (Ray Serve + FastAPI)
│   │   ├── serve.py                    # FastAPI app with prediction endpoints
│   │   ├── schemas.py                  # Pydantic request/response models
│   │   └── config.py                   # Serving configuration
│   └── _utils/                         # Shared utilities
│       └── logging.py                  # Rich console logging setup
├── tests/                              # Unit and integration tests
├── .devcontainer/                      # VS Code DevContainer config
├── .github/workflows/                  # CI/CD workflows
│   ├── ci-code-quality.yaml            # SHared code quality
│   ├── ci-docker-build-push.yaml       # Shared docker image building
│   └── train.yaml                      # Automated training pipeline
├── Dockerfile                          # Multi-stage container build
├── pyproject.toml                      # Project dependencies and config
├── uv.lock                             # Dependency lock file
└── README.md                           # This file
```

______________________________________________________________________

<h2 id="contributing">👥 Contributing</h2>

Contributions are welcome! This project follows OpenCloudHub's contribution standards.

Please see our [Contributing Guidelines](https://github.com/opencloudhub/.github/blob/main/.github/CONTRIBUTING.md) and [Code of Conduct](https://github.com/opencloudhub/.github/main/.github/CODE_OF_CONDUCT.md) for more details.

______________________________________________________________________

<h2 id="license">📄 License</h2>

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for more information.

______________________________________________________________________

<h2 id="contact">📬 Contact</h2>

Organization Link: [https://github.com/OpenCloudHub](https://github.com/OpenCloudHub)

Project Link: [https://github.com/opencloudhub/ai-dl-bert](https://github.com/opencloudhub/ai-dl-bert)

______________________________________________________________________

<h2 id="acknowledgements">🙏 Acknowledgements</h2>

- [PyTorch Lightning](https://lightning.ai/) - Structured PyTorch training framework
- [MLflow](https://mlflow.org/) - ML lifecycle and model registry management
- [Ray](https://ray.io/) - Distributed computing, tuning, and serving
- [DVC](https://dvc.org/) - Data version control and reproducibility
- [UV](https://github.com/astral-sh/uv) - Fast Python package and project manager
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Pre-trained BERT models

<p align="right">(<a href="#readme-top">back to top</a>)</p>
