# ATC AI RPO — Cloud API
# Runs atc_api_server.py on a persistent container (Railway / Render / Fly.io)

FROM python:3.11-slim

# System deps for audio libs and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install CPU-only torch (smaller image; swap URL for GPU builds)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
# Exclude hardware-only packages that won't work in cloud
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy source
COPY code/     ./code/
COPY data/     ./data/
# Model dir: mount at runtime or bake in if size allows
# COPY models/   ./models/

# Expose API port
EXPOSE 8766

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8766/api/health')"

CMD ["python", "code/atc_api_server.py"]
