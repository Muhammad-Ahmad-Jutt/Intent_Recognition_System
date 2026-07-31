from dotenv import load_dotenv
import os
load_dotenv()
import argparse
import sys
from data_loader_trainer import DataLoaderTrainer
from unseen_data import UnseenDataTests
from pathlib import Path

class symlink_manager:
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
    def create_symlink_symlink( current_dir, model_dir):
        print(model_dir, current_dir)
        current_dir = Path(current_dir)
        model_dir = Path(model_dir)
        symlink_dir = current_dir/'current_model'
        if model_dir.is_relative_to(current_dir):
            if symlink_dir.exists() or symlink_dir.is_symlink():
                symlink_dir.unlink()
            relative_target = model_dir.relative_to(current_dir)
            symlink_dir.symlink_to(relative_target, target_is_directory=True)
            print("Created:", symlink_dir)
            print("Exists:", symlink_dir.exists())
            print("Is symlink:", symlink_dir.is_symlink())
            print("Resolves to:", symlink_dir.resolve())
            return symlink_dir
        else:
            return 'The two paths are not same'
if __name__ == "__main__":
    model_dir = None
    model_name = os.getenv("MODEL_NAME")
    dataset_path = os.getenv("dataset_path")
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
    symlink_path = os.getenv("symlinkpath")
    model_out_directory = os.getenv("model_out_directory")
    greater_is_better = os.getenv("greater_is_better") == "True"
    metrics_output_file = os.getenv("metrics_output_file")
    hf_api_token = os.getenv("HF_TOKEN")
    label_mapping_file_path = os.getenv("label_mapping_file_path")
    accuracy_comparison_file = os.getenv('accuracy_comparison_file')
    unseen_data_path = os.getenv('unseen_data_path')
    # creating symlinks for dataset, current model_path, new_model_path
    symlink_obj = symlink_manager()
    current_dir = symlink_manager.get_current_path()
    dataset_path = symlink_manager.join_path(current_dir, dataset_path)
    print(current_dir, symlink_path)
    symlink_path = symlink_manager.join_path(current_dir, symlink_path)
    symlink_manager.exist_or_create(symlink_path)
    model_out_directory = symlink_manager.join_path(current_dir, model_out_directory)
    symlink_manager.exist_or_create(model_out_directory)
    label_mapping_file_path = symlink_manager.join_path(current_dir, label_mapping_file_path)
    accuracy_comparison_file = symlink_manager.join_path(current_dir, accuracy_comparison_file)
    accuracy_comparison_file = symlink_manager.check_if_file_exist_or_create(accuracy_comparison_file)
    unseen_data_path = symlink_manager.join_path(current_dir, unseen_data_path)
    parser = argparse.ArgumentParser("train_or_fine_tune")
    parser.add_argument("train_or_fine_tune", help="Just write 'train' or 'fine-tune' as your requirement", type=str)
    args = parser.parse_args()
    print("You have provided the argument:", args.train_or_fine_tune)
    if args.train_or_fine_tune not in ["train", "fine-tune", "test"]:
        print("Please provide a valid argument: 'train' or 'fine-tune'")
        sys.exit(1)
    elif args.train_or_fine_tune == "train" or args.train_or_fine_tune == "fine-tune":
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
            symlink_path=symlink_path,
            greater_is_better=greater_is_better,
            metrics_output_file=metrics_output_file,
            hf_api_token=hf_api_token,
            label_mapping_file_path=label_mapping_file_path
        )
        result, model_dir = trainer.train_command(args.train_or_fine_tune)
        print('model trained successfull ')
        symlink_manager.create_symlink_symlink(current_dir,model_dir)
        # it must return the current path of new model
    elif args.train_or_fine_tune == 'test':
        unseen_data_obj = UnseenDataTests()
        acc_percentage = unseen_data_obj.main(unseen_data_path, symlink_path)
        decision,model_dir = unseen_data_obj.read_and_decide(accuracy_comparison_file, acc_percentage, symlink_path)
        if decision==True:
            print('dicision is true')
            symlink_manager.create_symlink_symlink(current_dir,model_dir)
        else:
            print('dicstion is false')
            sys.exit(1)
        
