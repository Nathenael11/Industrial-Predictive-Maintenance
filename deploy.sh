#!/usr/bin/env bash
# ==============================================================================
# Elevvo Predictive Maintenance EC2 Deployment Script
# ==============================================================================

set -eo pipefail

APP_NAME="elevvo_predictive_maintenance"
PORT=5000

echo "[1/5] Checking Docker & Environment..."
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose
    sudo systemctl start docker
    sudo systemctl enable docker
fi

echo "[2/5] Verifying Model Artifacts..."
if [ ! -f "model_artifacts/model.pkl" ]; then
    echo "Model artifact missing. Running model training pipeline..."
    python3 train_model.py
fi

echo "[3/5] Building Docker Image..."
docker build -t ${APP_NAME}:latest .

echo "[4/5] Running Container..."
if [ $(docker ps -aq -f name=${APP_NAME}) ]; then
    echo "Stopping existing container..."
    docker stop ${APP_NAME} || true
    docker rm ${APP_NAME} || true
fi

docker run -d \
  --name ${APP_NAME} \
  -p ${PORT}:5000 \
  --restart unless-stopped \
  ${APP_NAME}:latest

echo "[5/5] Performing Health Check..."
sleep 5

HEALTH_STATUS=$(curl -s http://localhost:${PORT}/health | grep '"status": "healthy"' || true)

if [ -n "$HEALTH_STATUS" ]; then
    echo "================================================="
    echo "SUCCESS: Predictive Maintenance Service Deployed!"
    echo "URL: http://localhost:${PORT}"
    echo "================================================="
else
    echo "ERROR: Health check failed. Checking container logs..."
    docker logs ${APP_NAME}
    exit 1
fi
