# ========================================
# ML Service Dockerfile
# GPU-accelerated TensorFlow Serving
# ========================================

# Base image with CUDA support
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 AS base

LABEL maintainer="Gait Analysis Team <team@gaitanalysis.com>"
LABEL description="GPU-accelerated ML inference service for gait analysis"
LABEL version="1.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# ========================================
# Python dependencies stage
# ========================================
FROM base AS python-deps

# Copy requirements
COPY backend/ml-service/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional ML libraries
RUN pip install --no-cache-dir \
    tensorflow==2.13.0 \
    tensorflow-serving-api==2.13.0 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    opencv-python-headless==4.8.0.74 \
    mediapipe==0.10.3 \
    numpy==1.24.3 \
    scipy==1.11.1 \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    pillow==10.0.0 \
    grpcio==1.56.2 \
    grpcio-tools==1.56.2 \
    prometheus-client==0.17.1 \
    uvicorn[standard]==0.23.2 \
    fastapi==0.101.1 \
    pydantic==2.1.1

# ========================================
# Model preparation stage
# ========================================
FROM python-deps AS model-prep

WORKDIR /app

# Copy model files
COPY assets/models/ ./models/
COPY backend/ml-service/src/model_utils.py ./src/

# Convert and optimize models for serving
RUN python src/model_utils.py --optimize-models --input-dir=./models --output-dir=./optimized-models

# ========================================
# Production stage
# ========================================
FROM python-deps AS production

# Create app user
RUN groupadd -r ml && useradd -r -g ml ml

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=ml:ml backend/ml-service/ .

# Copy optimized models
COPY --from=model-prep --chown=ml:ml /app/optimized-models/ ./models/

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp /app/cache && \
    chown -R ml:ml /app

# Install additional dependencies if needed
COPY backend/ml-service/requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Switch to non-root user
USER ml

# Expose ports
EXPOSE 8080 8081 9090

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Environment variables
ENV MODEL_PATH=/app/models
ENV BATCH_SIZE=8
ENV MAX_WORKERS=4
ENV GPU_MEMORY_FRACTION=0.8
ENV CACHE_SIZE=1000
ENV LOG_LEVEL=INFO

# Start command
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]

# ========================================
# GPU-specific configuration
# ========================================
# Enable GPU support
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# TensorFlow GPU configuration
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_MEMORY_FRACTION=0.8

# Labels for metadata
ARG BUILD_VERSION=unknown
ARG BUILD_DATE=unknown
ARG GIT_COMMIT=unknown

LABEL org.opencontainers.image.version=$BUILD_VERSION
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$GIT_COMMIT
LABEL org.opencontainers.image.source="https://github.com/gait-analysis/gait-analysis-pro"
LABEL org.opencontainers.image.title="Gait Analysis ML Service"
LABEL org.opencontainers.image.description="GPU-accelerated machine learning inference service for pose estimation and gait analysis"