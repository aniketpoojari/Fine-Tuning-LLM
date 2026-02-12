import argparse
from src.common import read_params
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

def tokenize(row, format, tokenizer):
    prompt = format["prompt"] + row[0] + "\n" + format["answer"]  # Changed from row.iloc[0]
    answer = row[1]  # Changed from row.iloc[1]

    prompt_tokens = tokenizer(prompt, truncation=False, add_special_tokens=False)
    answer_tokens = tokenizer(answer, truncation=False, add_special_tokens=False)

    input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"] + [tokenizer.eos_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def padding(row, MAX_LENGTH, tokenizer):
    padding_length = MAX_LENGTH - len(row["input_ids"])

    input_ids = row["input_ids"] + [tokenizer.pad_token_id] * padding_length
    attention_mask = row["attention_mask"] + [0] * padding_length
    labels = row["labels"] + [-100] * padding_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def preprocess_data(config_path):
    config = read_params(config_path)
    task_1_dataset_path = config["data"]["task_1_dataset_path"]
    task_2_dataset_path = config["data"]["task_2_dataset_path"]
    base_model = config["training"]["base_model"]
    task_1_dataset_processed_path = config["data"]["task_1_dataset_processed_path"]
    task_2_dataset_processed_path = config["data"]["task_2_dataset_processed_path"]
    
    # READ DATA
    task_1_df = pd.read_csv(task_1_dataset_path)
    task_2_df = pd.read_csv(task_2_dataset_path)

    # LAMBDA FUNCTIONS
    task_1_format = { "prompt": "Generate code: ", "answer": "Code: "}
    task_2_format = { "prompt": "Generate docstring: ", "answer": "Docstring: "}
    
    # INITIALIZE TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # FORMAT DATA - Fix the apply function
    task_1_tokenized = []
    for _, row in task_1_df.iterrows():
        tokenized_row = tokenize([row['text'], row['code']], task_1_format, tokenizer)
        task_1_tokenized.append(tokenized_row)
    
    task_2_tokenized = []
    for _, row in task_2_df.iterrows():
        tokenized_row = tokenize([row['code'], row['docstring']], task_2_format, tokenizer)
        task_2_tokenized.append(tokenized_row)
    
    # CONVERT TO DATAFRAME
    task_1_df = pd.DataFrame(task_1_tokenized)
    task_2_df = pd.DataFrame(task_2_tokenized)

    # FIND MAX LENGTH OF INPUT_IDS
    task_1_max_length = task_1_df["input_ids"].apply(len).max()
    task_2_max_length = task_2_df["input_ids"].apply(len).max()
    
    # PADDING
    task_1_padded = []
    for _, row in task_1_df.iterrows():
        padded_row = padding(row, task_1_max_length, tokenizer)
        task_1_padded.append(padded_row)
    
    task_2_padded = []
    for _, row in task_2_df.iterrows():
        padded_row = padding(row, task_2_max_length, tokenizer)
        task_2_padded.append(padded_row)

    # CONVERT TO DATAFRAME
    task_1_df = pd.DataFrame(task_1_padded)
    task_2_df = pd.DataFrame(task_2_padded)
     
    # SAVE DATA
    task_1_hf = Dataset.from_pandas(task_1_df)
    task_2_hf = Dataset.from_pandas(task_2_df)

    task_1_hf.save_to_disk(task_1_dataset_processed_path)
    task_2_hf.save_to_disk(task_2_dataset_processed_path)


if __name__ == "__main__":
    # PARSE CONFIG FILE ARGUMENT
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    
    # START DOWNLOADING
    preprocess_data(config_path=parsed_args.config)