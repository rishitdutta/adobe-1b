# syntax=docker/dockerfile:1
###############################################################################
# Stage 1 ─ builder: install deps + cache all models so run-time is offline   #
###############################################################################
FROM python:3.10-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/opt/hf_cache
ENV TRANSFORMERS_CACHE=$HF_HOME
ENV HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# ---------- system packages needed to compile wheels & run PyMuPDF ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python requirements (cached layer) ----------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------- copy project code for model loading -----------------------
COPY models.py .

# ---------- preload all HF models (cached layer) -------------
# This layer is only rebuilt if models.py or requirements.txt changes.
RUN python -c "from models import load_models; load_models(); print('✅ All models cached.')"


###############################################################################
# Stage 2 ─ slim runtime image (only what we need to execute)                 #
###############################################################################
FROM python:3.10-slim

# offline & telemetry-free
ENV HF_HOME=/opt/hf_cache
ENV TRANSFORMERS_CACHE=$HF_HOME
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# runtime OS deps (PyMuPDF needs libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy Python site-packages and cached models from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /opt/hf_cache /opt/hf_cache

# Copy the rest of your project code (changes to these won't trigger a redownload)
COPY . /app

# Default command: execute the pipeline
CMD ["python", "main.py"]
