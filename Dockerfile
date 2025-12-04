# syntax=docker/dockerfile:1
# =============================================================================
# Dockerfile - Multi-Stage Build for Training, Serving, and Development
# =============================================================================
#
# Purpose:
#   Optimized Docker images for different ML lifecycle stages:
#   - dev: Full environment with all dependencies (DevContainer)
#   - training: Ray-compatible image for distributed training
#   - serving: Lightweight image for production inference
#
# Why Multi-Stage?
#   - Shared base layer reduces build time and storage
#   - Each target includes only necessary dependencies
#   - Smaller production images
#
# Build:
#   docker build --target dev -t emotion:dev .
#   docker build --target training -t emotion:train .
#   docker build --target serving -t emotion:serve .
#
# =============================================================================

#==============================================================================#
# Build arguments
#==============================================================================#
ARG RAY_VERSION=2.48.0
ARG PYTHON_MAJOR=3
ARG PYTHON_MINOR=12
ARG DISTRO=bookworm
ARG RAY_PY_TAG=py${PYTHON_MAJOR}${PYTHON_MINOR}
ARG UV_PY_TAG=python${PYTHON_MAJOR}.${PYTHON_MINOR}-${DISTRO}

#==============================================================================#
# Stage: Base with UV + Core Dependencies (SHARED LAYER)
# -----------------------------------------------------------------------------
# This stage creates a shared foundation with:
# - UV package manager for fast, reproducible installs
# - Core dependencies from pyproject.toml (no extras)
# - Compiled bytecode for faster startup
#==============================================================================#
FROM ghcr.io/astral-sh/uv:${UV_PY_TAG} AS uv_base
WORKDIR /workspace/project

# Install minimal build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# UV configuration for optimized builds
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy dependency files for layer caching
COPY pyproject.toml uv.lock ./

# Install base dependencies (creates shared .venv)
# Uses cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project

#==============================================================================#
# Stage: Development (for devcontainer)
# -----------------------------------------------------------------------------
# Full development environment with:
# - All dependencies including dev tools
# - All optional extras (training + serving)
# - Source code mounted/copied
#==============================================================================#
FROM uv_base AS dev

# Install ALL dependencies including dev + all extras
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --no-install-project

COPY src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project" \
    ENVIRONMENT=development

#==============================================================================#
# Stage: TRAINING (production training image)
# -----------------------------------------------------------------------------
# Optimized for distributed training:
# - Based on official Ray image for cluster compatibility
# - Training-specific dependencies (PyTorch, Lightning, etc.)
# - Runs as non-root 'ray' user
#==============================================================================#
FROM rayproject/ray:${RAY_VERSION}-${RAY_PY_TAG} AS training
WORKDIR /workspace/project

USER ray

# Copy UV from base for consistent package management
COPY --from=uv_base /usr/local/bin/uv /usr/local/bin/uv
COPY --chown=ray:ray pyproject.toml uv.lock ./

# Copy shared .venv from uv_base (reuses cached dependencies)
COPY --from=uv_base --chown=ray:ray /workspace/project/.venv /workspace/project/.venv

# Add only training extras on top of base dependencies
RUN --mount=type=cache,target=/home/ray/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra training --no-dev --no-install-project

COPY --chown=ray:ray src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project" \
    ENVIRONMENT=training

#==============================================================================#
# Stage: SERVING (production serving image)
# -----------------------------------------------------------------------------
# Lightweight image for inference:
# - Based on Python slim for minimal size
# - Serving-specific dependencies only (FastAPI, Ray Serve)
# - Runs as non-root 'ray' user for security
#==============================================================================#
FROM python:3.12-slim-bookworm AS serving
WORKDIR /workspace/project

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -g 1000 ray && \
    useradd -m -u 1000 -g 1000 -s /bin/bash ray && \
    chown -R ray:ray /workspace/project

USER ray

# Copy UV from base for consistent package management
COPY --from=uv_base /usr/local/bin/uv /usr/local/bin/uv
COPY --chown=ray:ray pyproject.toml uv.lock ./

# Copy shared .venv from uv_base (reuses cached dependencies)
COPY --from=uv_base --chown=ray:ray /workspace/project/.venv /workspace/project/.venv

# Add only serving extras on top of base dependencies
RUN --mount=type=cache,target=/home/ray/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra serving --no-dev --no-install-project

COPY --chown=ray:ray src/ ./src/

ENV VIRTUAL_ENV="/workspace/project/.venv" \
    PATH="/workspace/project/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/project" \
    ENVIRONMENT=production
