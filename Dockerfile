
# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /var/cache/model

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Set environment variables
ENV MODEL_CACHE_DIR=/var/cache/model
ENV PORT=8501

# Copy and make run.sh executable
COPY run.sh .
RUN chmod +x run.sh

# Command to run the application
CMD ["./run.sh"]