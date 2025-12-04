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
    DistilBERT emotion classification with MLOps pipeline featuring PyTorch Lightning, MLflow tracking, Ray Tune hyperparameter optimization, and Ray Serve deployment.<br />
    <a href="https://github.com/opencloudhub"><strong>Explore OpenCloudHub »</strong></a>
  </p>
</div>

______________________________________________________________________

<details>
  <summary>📑 Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#thesis-context">Thesis Context</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

______________________________________________________________________

<h2 id="about">🎯 About</h2>

This repository demonstrates a production-ready MLOps pipeline for text classification using emotion detection as the example use case. It showcases the integration of modern ML tooling including **PyTorch Lightning** for structured training, **MLflow** for experiment tracking and model registry, **Ray Tune** for distributed hyperparameter optimization, and **Ray Serve** for scalable model deployment.

The project fine-tunes **DistilBERT** (a distilled version of BERT that retains 97% of BERT's performance while being 60% smaller and 60% faster) on emotion-labeled text data from the [Hugging Face emotions dataset](https://huggingface.co/datasets/dair-ai/emotion). Dataset versioning is managed through a centralized [DVC Registry](https://github.com/OpenCloudHub/data-registry), enabling reproducible training across environments.

This serves as a reference implementation for the **OpenCloudHub** project, demonstrating best practices for ML workflows that can scale from local development to production Kubernetes clusters.

______________________________________________________________________

<h2 id="thesis-context">📚 Thesis Context</h2>

This repository is part of a master's thesis project exploring **MLOps practices for deep learning workflows**. It serves as a practical demonstration of:

### Research Focus Areas

| Area | Implementation in This Repo |
|------|----------------------------|
| **Experiment Tracking** | MLflow with parent-child run hierarchy for hyperparameter tuning |
| **Model Versioning** | MLflow Model Registry with automatic registration from best trials |
| **Data Versioning** | DVC integration with external data registry for reproducibility |
| **Distributed Training** | Ray Tune + PyTorch Lightning for scalable hyperparameter search |
| **Model Serving** | Ray Serve with hot-reload capability for zero-downtime updates |
| **CI/CD Integration** | GitHub Actions triggering Argo Workflows on Kubernetes |

### Key Technical Contributions

1. **Unified MLflow Run Hierarchy**: Demonstrates organizing hyperparameter search trials as nested child runs under a parent run, enabling cleaner experiment comparison and model lineage tracking.

2. **Ray Tune + Lightning Integration**: Shows how to properly integrate Ray Tune's distributed trials with PyTorch Lightning's training loop while maintaining MLflow logging consistency.

3. **Hot Model Reloading**: Implements Ray Serve's `reconfigure` pattern for updating deployed models without service restart, preserving inference availability.

4. **Multi-Stage Docker Builds**: Uses shared base layers across training and serving images to reduce build times and storage while maintaining separation of concerns.

### Related Thesis Repositories

- [`OpenCloudHub/data-registry`](https://github.com/OpenCloudHub/data-registry) - Centralized DVC data versioning
- [`OpenCloudHub/infra-kubernetes`](https://github.com/OpenCloudHub/infra-kubernetes) - Kubernetes cluster infrastructure
- [`OpenCloudHub/local-compose-stack`](https://github.com/OpenCloudHub/local-compose-stack) - Local Docker Compose dev stack (MLflow, MinIO)
- [`OpenCloudHub/.github`](https://github.com/OpenCloudHub/.github) - Shared CI/CD workflows and Argo templates

______________________________________________________________________

**Key Learning Points:**

- Integration of PyTorch Lightning with MLflow for experiment tracking
- Parent-child MLflow runs for hyperparameter search organization
- Ray Tune callbacks for distributed training with automatic checkpointing
- Model registration and versioning in MLflow Model Registry
- Zero-downtime model serving with Ray Serve's `reconfigure` method
- DVC integration for reproducible dataset versioning
- Multi-stage Docker builds for optimized training and serving images

______________________________________________________________________

<h2 id="features">✨ Features</h2>

### MLOps Pipeline

- 🔬 **Experiment Tracking**: MLflow integration with parent-child run hierarchy for organized hyperparameter tuning experiments
- 📊 **Model Registry**: Automatic model registration with versioning, metadata, and label mappings
- 🎯 **Hyperparameter Optimization**: Ray Tune for distributed hyperparameter search with configurable search spaces
- ⚡ **Distributed Training**: Ray + PyTorch Lightning integration for scalable model training
- 🚀 **Model Serving**: FastAPI + Ray Serve for production inference with hot model reloading

### Development & Deployment

- 🐳 **Containerized Environment**: Multi-stage Docker builds with UV package manager for fast, reproducible builds
- 📦 **Data Versioning**: DVC integration with external data registry for reproducible dataset management
- 🧪 **VS Code DevContainer**: Pre-configured development environment with all tools ready
- 🔄 **CI/CD Ready**: GitHub Actions workflows triggering Argo MLOps pipelines on Kubernetes

### Key Integrations

#### PyTorch Lightning + MLflow

- **Automatic metric logging**: Loss, accuracy, and F1 score logged per epoch
- **Model checkpointing**: Best model checkpoint integrated with Ray Tune's `TuneReportCheckpointCallback`
- **Hyperparameter tracking**: All training hyperparameters logged for experiment comparison

#### Ray Tune + MLflow

- **Nested run hierarchy**: Parent run for the search, child runs for each trial
- **Distributed trials**: Parallel hyperparameter exploration with configurable resources (CPU/GPU)
- **Best model selection**: Automatic identification and registration of the best performing model

#### Ray Serve + MLflow

- **Model Registry integration**: Load models directly via `models:/model-name/version` URIs
- **Hot model updates**: Use `reconfigure` to update models without service restart
- **Model metadata API**: `/info` endpoint exposes model version, training info, and emotion labels

______________________________________________________________________

<h2 id="architecture">🏗️ Architecture</h2>

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DVC Registry (github.com/OpenCloudHub/data-registry)               │    │
│  │  └── data/emotion/                                                  │    │
│  │      ├── processed/train/train.parquet                              │    │
│  │      ├── processed/val/val.parquet                                  │    │
│  │      └── metadata.json (class labels, statistics)                   │    │
│  │  Tags: emotion-v0.3.0, emotion-v1.0.0                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ dvc.api.open(rev="emotion-v1.0.0")
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Ray Tune Hyperparameter Search                                     │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  MLflow Parent Run: "hyperparameter_search"                   │  │    │
│  │  │  ├─ Trial 1 (Child): lr=2e-5, batch=16  → val_loss=0.42       │  │    │
│  │  │  ├─ Trial 2 (Child): lr=5e-5, batch=32  → val_loss=0.38 ★    │  │    │
│  │  │  ├─ Trial 3 (Child): lr=3e-5, batch=16  → val_loss=0.45       │  │    │
│  │  │  └─ Trial 4 (Child): lr=4e-5, batch=32  → val_loss=0.41       │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  PyTorch Lightning Module (DistilBERT)                              │    │
│  │  ├─ Forward: input_ids → DistilBERT → logits → softmax              │    │
│  │  ├─ Metrics: loss, accuracy, F1 (logged to MLflow)                  │    │
│  │  └─ Optimizer: AdamW with weight decay                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ Best trial → mlflow.pytorch.log_model()
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        REGISTRY LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  MLflow Model Registry                                              │    │
│  │  models:/ci.emotion-classifier/1                                    │    │
│  │  ├─ model/  (PyTorch Lightning checkpoint)                          │    │
│  │  ├─ labels.json  (emotion class mappings)                           │    │
│  │  └─ MLmodel  (signature, requirements, metadata)                    │    │
│  │                                                                     │    │
│  │  Tags: dvc_data_version, docker_image_tag, argo_workflow_uid        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ mlflow.pytorch.load_model(model_uri)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SERVING LAYER                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Ray Serve + FastAPI                                                │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │  GET  /          → Service info, docs link                    │  │    │
│  │  │  GET  /health    → Status, uptime, model loaded               │  │    │
│  │  │  GET  /info      → Model version, labels, training metadata   │  │    │
│  │  │  POST /predict   → Batch emotion classification               │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  Hot Reload: reconfigure({"model_uri": "models:/name/2"})           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Docker Multi-Stage Build

```
┌─────────────────────────────────────────────────────────────┐
│                    uv_base (shared layer)                   │
│  - UV package manager + core dependencies                   │
│  - Compiled bytecode for fast startup                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│     dev     │     │   training  │     │   serving   │
│  + all deps │     │  + ray base │     │  + slim img │
│  + dev tools│     │  + lightning│     │  + fastapi  │
│  DevContainer│    │  + dvc      │     │  + ray serve│
└─────────────┘     └─────────────┘     └─────────────┘
```

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

2. **Open in DevContainer** (Recommended)

   VSCode: `Ctrl+Shift+P` → `Dev Containers: Rebuild and Reopen in Container`

   Or **setup locally without DevContainer**:

   ```bash
   # Install UV
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync --all-extras
   ```

3. **Start infrastructure** (choose one option)

   **Option A: Local Docker Compose Stack** (quick testing)
   
   Use the [local-compose-stack](https://github.com/OpenCloudHub/local-compose-stack) for a quick MLflow + MinIO setup:
   
   ```bash
   # In a separate directory
   git clone https://github.com/OpenCloudHub/local-compose-stack.git
   cd local-compose-stack
   docker compose up -d
   ```
   
   Then configure this project:
   ```bash
   # Back in ai-dl-bert
   set -a && source .env.docker && set +a
   ```
   
   Access UIs:
   - MLflow: `http://localhost:5000`
   - MinIO Console: `http://localhost:9001`

   **Option B: Minikube** (closer to production)
   
   Connect to your running minikube cluster with services from [infra-kubernetes](https://github.com/OpenCloudHub/infra-kubernetes):
   
   ```bash
   set -a && source .env.minikube && set +a
   ```
   
   Access UIs via ingress:
   - MLflow: `https://mlflow.internal.opencloudhub.org`
   - MinIO: `https://minio.internal.opencloudhub.org`

4. **Start local Ray cluster** (for local training)

   ```bash
   ray start --head --num-cpus 8
   ```

   Access Ray dashboard at `http://127.0.0.1:8265`

You're now ready to train and serve models!

______________________________________________________________________

<h2 id="configuration">⚙️ Configuration</h2>

### Environment Files

Two pre-configured environment files are provided:

| File | Use Case | Infrastructure |
|------|----------|----------------|
| `.env.docker` | Local development | [local-compose-stack](https://github.com/OpenCloudHub/local-compose-stack) |
| `.env.minikube` | Minikube testing | [infra-kubernetes](https://github.com/OpenCloudHub/infra-kubernetes) |

**Apply configuration:**
```bash
set -a && source .env.docker && set +a    # For Docker Compose
set -a && source .env.minikube && set +a  # For Minikube
```

### Environment Variables

The application uses Pydantic Settings for type-safe configuration:

#### Training Configuration (`src/training/config.py`)

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | *Required* |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name in MLflow | `emotion-classification` |
| `MLFLOW_REGISTERED_MODEL_NAME` | Model registry name | `emotion-classifier` |
| `DVC_DATA_VERSION` | DVC tag for data version | *Required* |
| `DVC_REPO` | DVC registry repository URL | `https://github.com/OpenCloudHub/data-registry` |
| `DVC_REMOTE` | DVC remote name (for S3/MinIO) | `None` |
| `ARGO_WORKFLOW_UID` | Argo workflow ID (for CI/CD) | `DEV` |
| `DOCKER_IMAGE_TAG` | Docker image tag (for CI/CD) | `DEV` |

#### Serving Configuration (`src/serving/config.py`)

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | HuggingFace model name for tokenizer | `distilbert-base-uncased` |
| `API_NAME` | API service name | `emotion-classifier` |
| `REQUEST_MAX_LENGTH` | Maximum input sequence length | `128` |

#### S3/MinIO Configuration (for artifact storage)

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | S3/MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | S3/MinIO secret key |
| `AWS_ENDPOINT_URL` | S3/MinIO endpoint URL |
| `MLFLOW_S3_ENDPOINT_URL` | MLflow S3 endpoint (same as above) |

______________________________________________________________________

<h2 id="usage">📖 Usage</h2>

### Hyperparameter Tuning

**Submit as Ray job:**

```bash
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit \
  --working-dir . \
  -- python src/training/tune.py --num-epochs 5 --num-samples 4 --limit 5000
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
│   ├── training/                       # Training and optimization module
│   │   ├── tune.py                     # Ray Tune + MLflow hyperparameter search
│   │   ├── data.py                     # DVC data loading and tokenization
│   │   ├── model.py                    # PyTorch Lightning DistilBERT module
│   │   └── config.py                   # Pydantic settings for training
│   ├── serving/                        # Model serving module
│   │   ├── serve.py                    # Ray Serve + FastAPI endpoints
│   │   ├── schemas.py                  # Pydantic request/response models
│   │   └── config.py                   # Pydantic settings for serving
│   └── _utils/                         # Shared utilities
│       └── logging.py                  # Rich console logging
├── notebooks/                          # Jupyter notebooks for exploration
├── .devcontainer/                      # VS Code DevContainer configuration
├── .github/workflows/                  # CI/CD workflows
│   ├── ci-code-quality.yaml            # Code quality checks (ruff)
│   ├── ci-docker-build-push.yaml       # Conditional Docker image builds
│   └── train.yaml                      # MLOps pipeline trigger
├── .env.docker                         # Config for local-compose-stack
├── .env.minikube                       # Config for minikube cluster
├── Dockerfile                          # Multi-stage build (dev/training/serving)
├── pyproject.toml                      # Project dependencies
├── uv.lock                             # Dependency lock file
└── README.md                           # This documentation
```

### Module Details

#### `src/training/` - Training Pipeline

- **`tune.py`**: Main entry point for hyperparameter search. Creates MLflow parent run, launches Ray Tune trials, and registers the best model.
- **`model.py`**: PyTorch Lightning `LightningModule` wrapping DistilBERT with multi-class metrics (accuracy, F1).
- **`data.py`**: Loads data from DVC registry, tokenizes with HuggingFace tokenizer, creates PyTorch datasets.
- **`config.py`**: Configuration classes using Pydantic Settings for environment variable binding.

#### `src/serving/` - Serving API

- **`serve.py`**: Ray Serve deployment with FastAPI ingress. Supports model loading from MLflow and hot updates via `reconfigure()`.
- **`schemas.py`**: Pydantic models for request/response validation (`PredictionRequest`, `PredictionResponse`, `ModelInfo`).
- **`config.py`**: Serving-specific configuration (model name, max length, API name).

#### `.github/workflows/` - CI/CD Pipelines

- **`ci-code-quality.yaml`**: Runs on PRs and non-main branches. Uses shared workflow for ruff linting.
- **`ci-docker-build-push.yaml`**: Runs on main branch. Builds training/serving images conditionally based on changed files.
- **`train.yaml`**: Manual dispatch workflow that triggers Argo MLOps pipeline on Kubernetes cluster.

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

______________________________________________________________________

<div align="center">
  <h3>🌟 Follow the Journey</h3>
  <p><em>Building in public • Learning together • Sharing knowledge</em></p>

<div>
    <a href="https://opencloudhub.github.io/docs">
      <img src="https://img.shields.io/badge/Read%20the%20Docs-2596BE?style=for-the-badge&logo=read-the-docs&logoColor=white" alt="Documentation">
    </a>
    <a href="https://github.com/orgs/opencloudhub/discussions">
      <img src="https://img.shields.io/badge/Join%20Discussion-181717?style=for-the-badge&logo=github&logoColor=white" alt="Discussions">
    </a>
    <a href="https://github.com/orgs/opencloudhub/projects/4">
      <img src="https://img.shields.io/badge/View%20Roadmap-0052CC?style=for-the-badge&logo=jira&logoColor=white" alt="Roadmap">
    </a>
  </div>
</div>