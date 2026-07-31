#!/bin/bash

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


MODEL_PATH="$CURRENT_DIR/$symlinkpath"

echo "Checking model: $MODEL_PATH"


if [ -L "$MODEL_PATH" ] && [ -e "$MODEL_PATH" ]; then
    echo "Model exists"
    echo "Starting fine-tuning..."

    python -u main.py fine-tune
    EXIT_CODE=$?

else
    echo "Model does not exist"
    echo "Starting initial training..."

    python -u main.py train
    EXIT_CODE=$?
fi


if [ $EXIT_CODE -eq 0 ]; then
    echo "Model training successful"
    echo "Evaluating and sending model to ECR..."
    python -u api.py
    # Add your ECR push command here
else
    echo "Model training failed or accuracy is not acceptable"
    echo "Stopping pipeline"

    exit 1
fi