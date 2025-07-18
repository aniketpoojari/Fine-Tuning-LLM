import argparse
from src.common import read_params
from datasets import load_dataset
import pandas as pd




def get_data(config_path):
    config = read_params(config_path)
    task_1_dataset_name = config["data"]["task_1_dataset_name"]
    task_1_dataset_path = config["data"]["task_1_dataset_path"]
    task_2_dataset_name = config["data"]["task_2_dataset_name"]
    task_2_dataset_language = config["data"]["task_2_dataset_language"]
    task_2_dataset_path = config["data"]["task_2_dataset_path"]

    ## TASK 1 DATASET
    task_1_dataset = load_dataset(task_1_dataset_name)
    task_2_dataset = load_dataset(task_2_dataset_name, task_2_dataset_language)

    # combine train and test and validation splits
    task_1_text = task_1_dataset["train"]['text'] + task_1_dataset["validation"]['text'] + task_1_dataset["test"]['text']
    task_1_code = task_1_dataset["train"]['code'] + task_1_dataset["validation"]['code'] + task_1_dataset["test"]['code']
    task_2_code = task_2_dataset["train"]['code'] + task_2_dataset["validation"]['code'] + task_2_dataset["test"]['code']
    task_2_docstring = task_2_dataset["train"]['docstring'] + task_2_dataset["validation"]['docstring'] + task_2_dataset["test"]['docstring']

    # make dataframe
    task_1_df = {"text": task_1_text, "code": task_1_code}
    task_1_df = pd.DataFrame(task_1_df)
    task_2_df = {"code": task_2_code, "docstring": task_2_docstring}
    task_2_df = pd.DataFrame(task_2_df)
    # get rows from task_2_df whose code+docstring split length is les than 600
    task_2_df = task_2_df[task_2_df.apply(lambda row: len(row["code"].split(" ")) + len(row["docstring"].split(" ")) < 600, axis=1)][:1000]

    # save dataframe to csv
    task_1_df.to_csv(task_1_dataset_path, index=False)
    task_2_df.to_csv(task_2_dataset_path, index=False)
    
if __name__ == "__main__":
    # PARSE CONFIG FILE ARGUMENT
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    
    # START DOWNLOADING
    get_data(config_path=parsed_args.config)