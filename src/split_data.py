import argparse
from src.common import read_params
from datasets import load_from_disk

from sklearn.model_selection import train_test_split

def split_data(config_path):
    config = read_params(config_path)
    task_1_dataset_processed_path = config["data"]["task_1_dataset_processed_path"]
    task_2_dataset_processed_path = config["data"]["task_2_dataset_processed_path"]
    task_1_split = config["data"]["task_1_split"]
    task_1_train_path = config["data"]["task_1_train_path"]
    task_1_val_path = config["data"]["task_1_val_path"]
    task_2_split = config["data"]["task_2_split"]
    task_2_train_path = config["data"]["task_2_train_path"]
    task_2_val_path = config["data"]["task_2_val_path"]

    # READ DATA
    task_1_dataset = load_from_disk(task_1_dataset_processed_path)
    task_2_dataset = load_from_disk(task_2_dataset_processed_path)
    
    # SPLIT DATA
    dataset = task_1_dataset.train_test_split(train_size=task_1_split, seed=42)
    task_1_train, task_1_val = dataset['train'], dataset['test']
    dataset = task_2_dataset.train_test_split(train_size=task_2_split, seed=42)
    task_2_train, task_2_val = dataset['train'], dataset['test']

    # WRITE DATA
    task_1_train.save_to_disk(task_1_train_path)
    task_1_val.save_to_disk(task_1_val_path)
    task_2_train.save_to_disk(task_2_train_path)
    task_2_val.save_to_disk(task_2_val_path)

if __name__ == "__main__":
    # PARSE CONFIG FILE ARGUMENT
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    
    # START DOWNLOADING
    split_data(config_path=parsed_args.config)