# Intent Recognition System

## Overview
This repository implements an intent recognition system for customer support text. It uses Hugging Face transformers to train and fine-tune a text classification model, evaluate it on unseen data, and expose a REST API for inference.

## Core Components

- `main.py`
  - Orchestrates training, fine-tuning, evaluation, and model deployment steps.
  - Loads configuration from environment variables.
  - Downloads the latest existing model, creates output folders, and executes the requested command.
  - Uploads model artifacts and metrics to S3 when a new model passes evaluation.

- `data_loader_trainer.py`
  - Loads and validates dataset CSV files.
  - Preprocesses text using `transformers` tokenizers.
  - Trains or fine-tunes a sequence classification model.
  - Saves model checkpoints, tokenizer files, and training metrics.

- `api.py`
  - Implements a Flask HTTP API endpoint at `/intent_recognition`.
  - Loads the model from `current_model/` and returns predicted label and confidence.

- `manage_s3_buckets.py`
  - Creates and manages an S3 bucket.
  - Uploads trained model directories and metadata files.
  - Downloads model folders and reads JSON metadata from S3.

- `unseen_data.py`
  - Evaluates the trained model on unseen JSON test data.
  - Computes accuracy against true intent labels.
  - Compares new model accuracy to the previous best and decides whether to update the stored model metadata.

## Data Sources

- Training data comes from the Kaggle Customer Support Intent Classification Dataset.
- Additional external evaluation data is provided in `external_evaluation_dataset.json`.
- Unseen evaluation data is stored in `unseen_data.json`.

## Deployment and API

- The service is deployed via Flask.
- The model is loaded from `current_model/` on startup.
- Expected request format:

```json
{
  "query": "Your customer support text here"
}
```

- Example response:

```json
{
  "query": "Your customer support text here",
  "predicted_class": "<intent_label>",
  "confidence": 0.95
}
```

## Workflow Summary

1. Configure environment variables in `.env` or use `env.example`.
2. Run `main.py` with `command=train` or `command=fine-tune`.
3. `main.py` downloads the latest model, loads data, and starts training.
4. The system saves the trained model and tokenizer to a timestamped output folder.
5. Trained artifacts are uploaded to S3.
6. The model is evaluated on unseen JSON data.
7. If the accuracy meets or exceeds the previous best, metadata is updated and uploaded.
8. The Flask API loads the latest `current_model/` for inference.

## Notes

- The project uses environment variables for model settings, S3 credentials, dataset paths, training hyperparameters, and output directories.
- The training code currently tracks accuracy via `evaluate.load("accuracy")` and saves metrics to JSON.
- S3 integration is handled with `boto3`.
- The model inference API requires a query longer than 5 characters.
