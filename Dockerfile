FROM python:3.9-slim

WORKDIR /app

# Install only essential build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY . .

# Create cache directory for model files
RUN mkdir -p /var/cache/model
ENV MODEL_CACHE_DIR=/var/cache/model

# Make port 8000 available
EXPOSE 8000

# Run with gunicorn using preload flag
CMD gunicorn api:app -k uvicorn.workers.UvicornWorker --preload --workers 2 --bind 0.0.0.0:${PORT:-8000}
