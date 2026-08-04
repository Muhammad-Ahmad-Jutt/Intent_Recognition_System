#!/bin/bash
set -e

ENVIRONMENT="${environment:-development}"

echo "========================================"
echo "Intent Recognition Startup"
echo "Environment : $ENVIRONMENT"
echo "========================================"

# ----------------------------
# Production Mode
# ----------------------------
if [ "$ENVIRONMENT" = "production" ]; then

    echo "Production environment detected."
    # python -u test.py 
    echo "Downloading latest model..."

    python -u download_model.py

    echo "Starting API..."

    exec python -u api.py

fi

# ----------------------------
# Development Mode
# ----------------------------
echo "Development environment detected."

python -u main.py 

EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "========================================"
    echo "Training completed successfully."
    echo "========================================"
    exit 0
else
    echo "========================================"
    echo "Training failed."
    echo "Pipeline stopped."
    echo "========================================"
    exit 1
fi