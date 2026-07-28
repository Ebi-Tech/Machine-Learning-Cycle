FROM python:3.11-slim

# PYTHONUNBUFFERED so logs show up right away instead of being buffered.
# TF_CPP_MIN_LOG_LEVEL=3 hides TensorFlow's noisy startup messages.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /app

# libgl1-mesa-glx was renamed to libgl1 in current Debian releases.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installing requirements before copying the rest of the app means this
# layer stays cached (and pip install gets skipped) on rebuilds where only
# the app code changed, not requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/retrain data/train data/test models

EXPOSE 5000

# Docker/compose use this to know when the container is ready, and can
# restart it automatically if /health fails 3 times in a row.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",5000)}/health')" || exit 1

# Gunicorn instead of Flask's dev server, which isn't built for real traffic.
# One worker to keep memory use predictable.
# No --preload: with only one worker there's no benefit to it, and
# preloading TensorFlow before gunicorn forks the worker causes a deadlock
# — a lock held by one of TensorFlow's background threads never gets
# released in the forked worker, since that thread doesn't exist there
# anymore. Each worker loads the model itself instead.
# Timeout 120 since retraining requests can take a while.
# Shell form (no brackets) so ${PORT:-5000} still works.
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 app:app
