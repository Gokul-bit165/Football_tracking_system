#!/bin/bash
# Start MLflow Tracking Server with shared sqlite backend and local artifact storage

# Ensure directory for mlflow db and artifacts exists
mkdir -p experiments/mlflow/artifacts

echo "Starting MLflow Tracking Server at http://0.0.0.0:5000"
echo "Artifacts directory: ./experiments/mlflow/artifacts"
echo "Database: sqlite:///experiments/mlflow/mlflow.db"

python -m mlflow server \
    --backend-store-uri sqlite:///experiments/mlflow/mlflow.db \
    --default-artifact-root ./experiments/mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000
