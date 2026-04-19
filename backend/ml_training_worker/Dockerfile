FROM python:3.11-slim

WORKDIR /app

# System packages needed for scikit-learn / catch22
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install all requirements unified for the ml training worker
COPY ml_training_worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application source needed at runtime
COPY ml_training_worker/ /app/ml_training_worker/
COPY ml_training/ /app/ml_training/
COPY goldenduck_core/ /app/goldenduck_core/
CMD ["python", "-m", "ml_training_worker.training_worker"]
