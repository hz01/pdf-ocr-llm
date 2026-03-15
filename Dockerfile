# PyTorch 2.10 + CUDA 12.8 (devel for building flash-attn; has nvcc)
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (--break-system-packages: safe in container, base image uses PEP 668)
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Flash Attention 2 (builds from source; devel image has nvcc)
# Flash Attention 3 (prebuilt wheel — cu128 + torch2.10, no nvcc needed)
RUN pip install --no-cache-dir --break-system-packages \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.8.2/flash_attn_3-3.0.0+cu128torch2.10gite2743ab-cp39-abi3-linux_x86_64.whl

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

# Expose any ports if needed (optional)
# EXPOSE 8000

# Expose UI and API ports
EXPOSE 7860 8000

# Default: serve UI + API (override in docker-compose if needed)
CMD ["python", "main.py", "serve"]

