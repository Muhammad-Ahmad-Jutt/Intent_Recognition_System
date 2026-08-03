# data _load er_trainer.py

from datetime import datetime
import re, json
import unicodedata
import numpy as np
import pandas as pd
import evaluate
import os
from sklearn.model_selection import train_test_split

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class DataLoaderTrainer:
    def __init__(self, model_name, dataset_path, test_size, random_state,num_train_epochs, per_device_train_batch_size,
                 per_device_eval_batch_size, weight_decay, eval_strategy, save_strategy, load_best_model_at_end,
                 logging_steps, learning_rate, 
                 logging_strategy, metric_for_best_model, directory_name,model_out_directory
                 , greater_is_better, metrics_output_file, hf_api_token, label_mapping_file_path):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state
        self.directory_name = directory_name
        self.model_out_directory = model_out_directory
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.weight_decay = weight_decay
        self.save_strategy = save_strategy
        self.load_best_model_at_end = load_best_model_at_end
        self.logging_steps = logging_steps
        self.learning_rate = learning_rate
        self.eval_strategy = eval_strategy
        self.logging_strategy = logging_strategy
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.metrics_output_file = metrics_output_file
        self.hf_api_token = hf_api_token
        self.label_mapping_file_path = label_mapping_file_path
        self.metric = evaluate.load("accuracy")
    def train_command(self, train_command):
        id2label, label2id = self.load_label_mapping()
        if train_command == "train":
            print("You have chosen to train the model from scratch.")
                
        # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)
            train_dataset, test_dataset = self.preprocess_datasets()
            results = self.train_model(train_dataset, test_dataset, self.model_out_directory)
            return results
        elif train_command == "fine-tune":
            print("You have chosen to fine-tune the model on your dataset.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.directory_name,id2label=id2label, label2id=label2id)

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.directory_name
            )
            train_dataset, test_dataset = self.preprocess_datasets()
            results = self.train_model(train_dataset, test_dataset, self.model_out_directory)
            return results
        else:
            raise ValueError("Please provide a valid argument: 'train' or 'fine-tune'")
    def load_label_mapping(self):

        with open(self.label_mapping_file_path, "r") as f:
            id2label = json.load(f)

        id2label = {int(k): v for k, v in id2label.items()}

        label2id = {
            v: k 
            for k, v in id2label.items()
        }

        return id2label, label2id
    def preprocess_datasets(self):
            # Preprocess the datasets
        train_dataset, test_dataset = self.load_dataset()
        train_dataset = train_dataset.map(lambda x: self.preprocess_function(x['cleaned_text']), batched=True)
        test_dataset = test_dataset.map(lambda x: self.preprocess_function(x['cleaned_text']), batched=True)
        return train_dataset, test_dataset
        
    def load_dataset(self):
        # Load the dataset from the CSV file
        df = pd.read_csv(self.dataset_path)

        # Check if the required columns are present
        if 'cleaned_text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'cleaned_text' and 'label' columns.")
        # Check if the 'label' column is present
        if 'label' not in df.columns:
            raise ValueError("Dataset must contain a 'label' column.")
        
        # Split the dataset into training and testing sets
        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)

        # Convert to Hugging Face Dataset format
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        return train_dataset, test_dataset
    def preprocess_function(self, text_column):

        return self.tokenizer(text_column, truncation=True, padding=False)
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)
    
    def train_model(self, train_dataset, test_dataset, output_dir):
            training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy=self.eval_strategy,
            save_strategy=self.save_strategy,
            logging_strategy=self.logging_strategy,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            logging_steps=self.logging_steps,
        )
            trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics,
        )
            trainer.train()
            direcory_path = self.create_new_directory(output_dir)
            trainer.save_model(direcory_path)
            self.tokenizer.save_pretrained(direcory_path)
            results = self.save_training_metrics(trainer, direcory_path)
            return results, direcory_path
    def create_new_directory(self, path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir = os.path.join(path, timestamp)

        os.makedirs(new_dir, exist_ok=True)

        return new_dir
    def save_training_metrics(self,trainer, output_dir):
        """
        Extract training/evaluation metrics from Hugging Face Trainer
        and save them into a JSON file.
        """

        logs = trainer.state.log_history

        metrics = {
            "epochs": [],
            "train_loss": [],
            "eval_loss": [],
            "accuracy": [],
            "f1": []
        }

        for log in logs:

            # Training loss
            if "loss" in log and "eval_loss" not in log:
                metrics["epochs"].append(log["epoch"])
                metrics["train_loss"].append(log["loss"])


            # Validation loss
            if "eval_loss" in log:
                metrics["eval_loss"].append(log["eval_loss"])


            # Accuracy
            if "eval_accuracy" in log:
                metrics["accuracy"].append(log["eval_accuracy"])


            # F1 score (if compute_metrics provides it)
            if "eval_f1" in log:
                metrics["f1"].append(log["eval_f1"])


        # Calculate final summary metrics

        summary = {

            "best_eval_loss": min(metrics["eval_loss"])
            if metrics["eval_loss"] else None,

            "final_train_loss": metrics["train_loss"][-1]
            if metrics["train_loss"] else None,

            "final_accuracy": metrics["accuracy"][-1]
            if metrics["accuracy"] else None,

            "final_f1": metrics["f1"][-1]
            if metrics["f1"] else None
        }


        metrics["summary"] = summary
        file_path = os.path.join(output_dir, self.metrics_output_file)

        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)


        print(f"Metrics saved to {file_path}")

        return metrics
    
