#!/bin/bash
set -e

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="$CURRENT_DIR/${symlinkpath}"
ENVIRONMENT="${environment:-development}"

echo "========================================"
echo "Intent Recognition Startup"
echo "Environment : $ENVIRONMENT"
echo "Model Path  : $MODEL_PATH"
echo "========================================"

# ----------------------------
# Production Mode
# ----------------------------
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Production environment detected."
    echo "Starting API..."

    exec python -u api.py
fi

# ----------------------------
# Development Mode
# ----------------------------
echo "Development environment detected."

if [ -L "$MODEL_PATH" ] && [ -e "$MODEL_PATH" ]; then
    echo "Existing model found."
    echo "Starting fine-tuning..."

    python -u main.py fine-tune
else
    echo "No existing model found."
    echo "Starting initial training..."

    python -u main.py train
fi

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