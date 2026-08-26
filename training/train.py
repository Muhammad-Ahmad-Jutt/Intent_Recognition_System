import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import evaluate

from sklearn.model_selection import train_test_split

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# ============================================================
# SAGEMAKER DIRECTORIES
# ============================================================

SM_CHANNEL_TRAIN = os.environ.get(
    "SM_CHANNEL_TRAIN",
    "/opt/ml/input/data/train"
)

SM_MODEL_DIR = os.environ.get(
    "SM_MODEL_DIR",
    "/opt/ml/model"
)

SM_OUTPUT_DATA_DIR = os.environ.get(
    "SM_OUTPUT_DATA_DIR",
    "/opt/ml/output/data"
)


# ============================================================
# TRAINING CLASS
# ============================================================

class SageMakerTrainer:

    def __init__(self, args):

        self.args = args

        self.model_name = args.model_name

        self.training_command = (
            args.training_command
        )

        self.test_size = args.test_size
        self.random_state = args.random_state

        self.num_train_epochs = (
            args.num_train_epochs
        )

        self.per_device_train_batch_size = (
            args.per_device_train_batch_size
        )

        self.per_device_eval_batch_size = (
            args.per_device_eval_batch_size
        )

        self.weight_decay = args.weight_decay

        self.learning_rate = args.learning_rate

        self.eval_strategy = args.eval_strategy

        self.save_strategy = args.save_strategy

        self.logging_strategy = (
            args.logging_strategy
        )

        self.logging_steps = args.logging_steps

        self.load_best_model_at_end = (
            args.load_best_model_at_end
        )

        self.metric_for_best_model = (
            args.metric_for_best_model
        )

        self.greater_is_better = (
            args.greater_is_better
        )

        self.label_mapping_file = (
            args.label_mapping_file
        )

        self.dataset_file = (
            args.dataset_file
        )

        self.metric = evaluate.load(
            "accuracy"
        )

    # ========================================================
    # FIND TRAINING FILE
    # ========================================================

    def get_dataset_path(self):

        path = (
            Path(SM_CHANNEL_TRAIN)
            / self.dataset_file
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Training dataset not found: {path}"
            )

        return path

    # ========================================================
    # FIND LABEL MAPPING
    # ========================================================

    def get_label_mapping_path(self):

        path = (
            Path(SM_CHANNEL_TRAIN)
            / self.label_mapping_file
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Label mapping not found: {path}"
            )

        return path

    # ========================================================
    # LOAD LABEL MAPPING
    # ========================================================

    def load_label_mapping(self):

        path = self.get_label_mapping_path()

        with open(path, "r") as file:

            id2label = json.load(file)

        id2label = {
            int(key): value
            for key, value in id2label.items()
        }

        label2id = {
            value: key
            for key, value in id2label.items()
        }

        return id2label, label2id

    # ========================================================
    # LOAD DATASET
    # ========================================================

    def load_dataset(self):

        dataset_path = (
            self.get_dataset_path()
        )

        print(
            f"Loading dataset from: "
            f"{dataset_path}"
        )

        df = pd.read_csv(
            dataset_path
        )

        required_columns = [
            "cleaned_text",
            "label"
        ]

        for column in required_columns:

            if column not in df.columns:

                raise ValueError(
                    f"Dataset must contain "
                    f"'{column}' column."
                )

        print(
            f"Dataset size: {len(df)}"
        )

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df["label"]
        )

        train_dataset = (
            Dataset.from_pandas(
                train_df,
                preserve_index=False
            )
        )

        test_dataset = (
            Dataset.from_pandas(
                test_df,
                preserve_index=False
            )
        )

        print(
            f"Training examples: "
            f"{len(train_dataset)}"
        )

        print(
            f"Evaluation examples: "
            f"{len(test_dataset)}"
        )

        return (
            train_dataset,
            test_dataset
        )

    # ========================================================
    # TOKENIZATION
    # ========================================================

    def preprocess_function(
        self,
        examples
    ):

        return self.tokenizer(
            examples["cleaned_text"],
            truncation=True,
            padding=False
        )

    # ========================================================
    # PREPROCESS DATASETS
    # ========================================================

    def preprocess_datasets(
        self,
        train_dataset,
        test_dataset
    ):

        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True
        )

        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True
        )

        return (
            train_dataset,
            test_dataset
        )

    # ========================================================
    # METRICS
    # ========================================================

    def compute_metrics(
        self,
        eval_prediction
    ):

        logits, labels = eval_prediction

        predictions = np.argmax(
            logits,
            axis=-1
        )

        return self.metric.compute(
            predictions=predictions,
            references=labels
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(
        self,
        id2label,
        label2id
    ):

        if self.training_command == "train":

            print(
                f"Loading base model: "
                f"{self.model_name}"
            )

            tokenizer = (
                AutoTokenizer.from_pretrained(
                    self.model_name
                )
            )

            model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    self.model_name,
                    num_labels=len(label2id),
                    id2label=id2label,
                    label2id=label2id
                )
            )

            return tokenizer, model

        elif self.training_command == "fine-tune":

            if not self.args.previous_model:

                raise ValueError(
                    "previous_model is required "
                    "when training_command=fine-tune"
                )

            print(
                "Loading previous model:"
            )

            print(
                self.args.previous_model
            )

            tokenizer = (
                AutoTokenizer.from_pretrained(
                    self.args.previous_model
                )
            )

            model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    self.args.previous_model
                )
            )

            return tokenizer, model

        else:

            raise ValueError(
                "training_command must be "
                "'train' or 'fine-tune'"
            )

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self):

        print("=" * 60)
        print("SAGEMAKER TRAINING JOB")
        print("=" * 60)

        print(
            f"Training command: "
            f"{self.training_command}"
        )

        print(
            f"Model: "
            f"{self.model_name}"
        )

        print(
            f"SageMaker model directory: "
            f"{SM_MODEL_DIR}"
        )

        # ----------------------------------------------------
        # Label mapping
        # ----------------------------------------------------

        id2label, label2id = (
            self.load_label_mapping()
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        train_dataset, test_dataset = (
            self.load_dataset()
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        (
            self.tokenizer,
            self.model
        ) = self.load_model(
            id2label,
            label2id
        )

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        (
            train_dataset,
            test_dataset
        ) = self.preprocess_datasets(
            train_dataset,
            test_dataset
        )

        # ----------------------------------------------------
        # Data collator
        # ----------------------------------------------------

        data_collator = (
            DataCollatorWithPadding(
                tokenizer=self.tokenizer
            )
        )

        # ----------------------------------------------------
        # Training arguments
        # ----------------------------------------------------

        training_args = TrainingArguments(

            output_dir=SM_MODEL_DIR,

            eval_strategy=self.eval_strategy,

            save_strategy=self.save_strategy,

            logging_strategy=self.logging_strategy,

            learning_rate=self.learning_rate,

            per_device_train_batch_size=(
                self.per_device_train_batch_size
            ),

            per_device_eval_batch_size=(
                self.per_device_eval_batch_size
            ),

            num_train_epochs=(
                self.num_train_epochs
            ),

            weight_decay=self.weight_decay,

            load_best_model_at_end=(
                self.load_best_model_at_end
            ),

            metric_for_best_model=(
                self.metric_for_best_model
            ),

            greater_is_better=(
                self.greater_is_better
            ),

            logging_steps=self.logging_steps,

            report_to="none"
        )

        # ----------------------------------------------------
        # Trainer
        # ----------------------------------------------------

        trainer = Trainer(

            model=self.model,

            args=training_args,

            train_dataset=train_dataset,

            eval_dataset=test_dataset,

            processing_class=self.tokenizer,

            data_collator=data_collator,

            compute_metrics=self.compute_metrics
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        print()
        print("Starting model training...")
        print()

        trainer.train()

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        print()
        print("Running final evaluation...")
        print()

        evaluation_metrics = (
            trainer.evaluate()
        )

        print(
            "Evaluation metrics:"
        )

        print(
            evaluation_metrics
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        print()
        print(
            f"Saving model to: "
            f"{SM_MODEL_DIR}"
        )

        trainer.save_model(
            SM_MODEL_DIR
        )

        self.tokenizer.save_pretrained(
            SM_MODEL_DIR
        )

        # ----------------------------------------------------
        # Save metrics
        # ----------------------------------------------------

        self.save_metrics(
            trainer,
            evaluation_metrics
        )

        print()
        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

    # ========================================================
    # SAVE METRICS
    # ========================================================

    def save_metrics(
        self,
        trainer,
        evaluation_metrics
    ):

        os.makedirs(
            SM_OUTPUT_DATA_DIR,
            exist_ok=True
        )

        metrics = {

            "training_command":
                self.training_command,

            "model_name":
                self.model_name,

            "timestamp":
                datetime.utcnow().isoformat(),

            "evaluation":
                evaluation_metrics,

            "training_log":
                trainer.state.log_history
        }

        metrics_file = (
            Path(SM_OUTPUT_DATA_DIR)
            / "training_metrics.json"
        )

        with open(
            metrics_file,
            "w"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4,
                default=str
            )

        print(
            f"Metrics saved to: "
            f"{metrics_file}"
        )


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased"
    )

    parser.add_argument(
        "--training-command",
        type=str,
        choices=[
            "train",
            "fine-tune"
        ],
        default="train"
    )

    parser.add_argument(
        "--previous-model",
        type=str,
        default=None
    )

    parser.add_argument(
        "--dataset-file",
        type=str,
        default="training.csv"
    )

    parser.add_argument(
        "--label-mapping-file",
        type=str,
        default="label_mapping.json"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42
    )

    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5
    )

    parser.add_argument(
        "--eval-strategy",
        type=str,
        default="epoch"
    )

    parser.add_argument(
        "--save-strategy",
        type=str,
        default="epoch"
    )

    parser.add_argument(
        "--logging-strategy",
        type=str,
        default="steps"
    )

    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50
    )

    parser.add_argument(
        "--load-best-model-at-end",
        action="store_true"
    )

    parser.add_argument(
        "--metric-for-best-model",
        type=str,
        default="accuracy"
    )

    parser.add_argument(
        "--greater-is-better",
        action="store_true"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    trainer = SageMakerTrainer(
        args
    )

    trainer.train()