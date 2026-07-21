FROM python:3.11-slim

# Set environment variables to prevent Python from buffering stdout/stderr
# and to prevent TensorFlow from printing GPU warnings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /app

# Install system dependencies needed by some Python packages.
# libgl1-mesa-glx was renamed to libgl1 in current Debian releases.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (Docker layer caching: if requirements.txt
# doesn't change, this layer is cached and pip install is skipped on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/retrain data/train data/test models

# Expose the port Flask runs on
EXPOSE 5000

# Health check: Docker and docker-compose use this to know if the container is ready.
# It hits /health every 30 seconds. If it fails 3 times in a row, the container is
# marked unhealthy and can be restarted automatically.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Run with gunicorn for production (not Flask's dev server).
# Gunicorn is a production WSGI server that handles concurrent requests properly.
# 4 workers means 4 parallel request handlers per container.
# Timeout 120 because retraining requests can take time.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
