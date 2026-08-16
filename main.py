from dotenv import load_dotenv
import os

from download_model import download_latest_model
from manage_s3_buckets import S3_bucket_crud
load_dotenv()
import argparse
import sys
from data_loader_trainer import DataLoaderTrainer
from unseen_data import UnseenDataTests
from pathlib import Path
class manage_paths:
    def exist_or_create(dir):
        dir = Path(dir)
        if dir.exists:
            return
        else:
            dir.mkdir()
            return
    def get_current_path():
        current_dir = Path(__file__).resolve().parent

        return current_dir
    def join_path(parent_path, child_path):
        return parent_path.joinpath(child_path)
    def check_if_file_exist_or_create(file_path):
        file_path = Path(file_path)
        if file_path.exists():
            return file_path
        else:
            open(file_path, 'x')
            return file_path
if __name__ == "__main__":
    model_dir = None
    model_name = os.getenv("MODEL_NAME")
    dataset_path = os.getenv("dataset_path")
    new_dataset_folder = os.getenv("new_dataset_folder")
    test_size = float(os.getenv("test_size"))
    random_state = int(os.getenv("random_state"))
    num_train_epochs = int(os.getenv("num_train_epochs"))
    per_device_train_batch_size = int(os.getenv("per_device_train_batch_size"))
    per_device_eval_batch_size = int(os.getenv("per_device_eval_batch_size"))
    weight_decay = float(os.getenv("weight_decay"))
    eval_strategy = os.getenv("eval_strategy")
    save_strategy = os.getenv("save_strategy")
    load_best_model_at_end = os.getenv("load_best_model_at_end") == "True"
    logging_steps = int(os.getenv("logging_steps"))
    learning_rate = float(os.getenv("learning_rate"))
    logging_strategy = os.getenv("logging_strategy")
    metric_for_best_model = os.getenv("metric_for_best_model")
    # model_path = os.getenv("model_path")
    model_out_directory = os.getenv("model_out_directory")
    greater_is_better = os.getenv("greater_is_better") == "True"
    metrics_output_file = os.getenv("metrics_output_file")
    hf_api_token = os.getenv("HF_TOKEN")
    label_mapping_file_path = os.getenv("label_mapping_file_path")
    accuracy_comparison_file = os.getenv('accuracy_comparison_file')
    unseen_data_path = os.getenv('unseen_data_path')
    s3_bucket_name = os.getenv('s3_bucket_name')
    aws_access_key_id = os.getenv('aws_access_key_id')
    aws_secret_access_key = os.getenv('aws_secret_access_key')
    aws_region_name = os.getenv('aws_region_name')
    aws_s3_obj = S3_bucket_crud(s3_bucket_name, aws_access_key_id, aws_secret_access_key, aws_region_name)
    aws_s3_obj.create_bucket_if_not_exists()
    # creating symlinks for dataset, current model_path, new_model_path
    symlink_obj = manage_paths()
    current_dir = manage_paths.get_current_path()
    dataset_path = manage_paths.join_path(current_dir, dataset_path)
    # new_dataset_folder = symlink_manager.join_path(current_dir, new_dataset_folder)
    # model_path = manage_paths.join_path(current_dir, model_path)
    # manage_paths.exist_or_create(model_path)
    model_out_directory = manage_paths.join_path(current_dir, model_out_directory)
    manage_paths.exist_or_create(model_out_directory)
    label_mapping_file_path = manage_paths.join_path(current_dir, label_mapping_file_path)
    # accuracy_comparison_file = manage_paths.join_path(current_dir, accuracy_comparison_file)
    # accuracy_comparison_file = manage_paths.check_if_file_exist_or_create(accuracy_comparison_file)
    unseen_data_path = manage_paths.join_path(current_dir, unseen_data_path)
    train_command = os.getenv("command")
    model_path = download_latest_model()
    if train_command not in ["train", "fine-tune", "test"]:
        print("Please provide a valid argument: 'train' or 'fine-tune' or 'test'")
        sys.exit(1)
    elif train_command == "train" or train_command == "fine-tune" or train_command == "test":
        if model_path is not None:
            print(f"Existing model found at {model_path}. Proceeding with fine-tuning.")
            train_command = "fine-tune"
        trainer = DataLoaderTrainer(
            model_name=model_name,
            dataset_path=dataset_path,
            test_size=test_size,
            random_state=random_state,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            weight_decay=weight_decay,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            logging_strategy=logging_strategy,
            metric_for_best_model=metric_for_best_model,
            model_out_directory=model_out_directory,
            directory_name=model_path,
            greater_is_better=greater_is_better,
            metrics_output_file=metrics_output_file,
            hf_api_token=hf_api_token,
            label_mapping_file_path=label_mapping_file_path
        )
        result, model_dir = trainer.train_command(train_command)
        aws_s3_obj.upload_folder(model_dir)
        testing_model_on_unseen_data = UnseenDataTests()
        acc_percentage = testing_model_on_unseen_data.main(unseen_data_path, model_dir)
        decision = testing_model_on_unseen_data.read_and_decide(accuracy_comparison_file, acc_percentage, model_dir)
        if decision:
            aws_s3_obj.upload_file_to_s3(accuracy_comparison_file)
            print(f"Training completed. Model saved to {model_dir}. Accuracy: {acc_percentage}%\n accuracy_comparison_file updated and uploaded to S3.")
        else:
            print(f"Training completed. Model saved to {model_dir}. Accuracy: {acc_percentage}%\n accuracy_comparison_file not updated as the new model's accuracy is lower than the previous one.")
            sys.exit(1)
    # elif train_command == "test":

    # elif args.train_or_fine_tune == 'test':
    #     unseen_data_obj = UnseenDataTests()
    #     acc_percentage = unseen_data_obj.main(unseen_data_path, symlink_path)
    #     decision,model_dir = unseen_data_obj.read_and_decide(accuracy_comparison_file, acc_percentage, symlink_path)
    #     if decision==True:
    #         print('dicision is true')
    #         symlink_manager.create_symlink_symlink(current_dir,model_dir)
    #     else:
    #         print('dicstion is false')
    #         sys.exit(1)
        
