# PyTorch 2.10 + CUDA 12.8 (runtime image)
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

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

