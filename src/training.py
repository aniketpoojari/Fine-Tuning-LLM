import argparse
import json
import subprocess
from datetime import datetime

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from time import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.common import read_params


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def train_function(model, tokenizer, train_dataset, eval_dataset, target_modules, adapter_name, config):
    training_cfg = config["training"]

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=int(training_cfg["lora_r"]),
        lora_alpha=int(training_cfg["lora_alpha"]),
        target_modules=target_modules,
        lora_dropout=float(training_cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=f"saved_models/adapters/{adapter_name}",

        per_device_train_batch_size=int(training_cfg["batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        num_train_epochs=int(training_cfg["num_epochs"]),

        logging_steps=20,
        save_steps=100,
        save_total_limit=1,

        eval_strategy="steps",
        eval_steps=100,

        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,

        report_to="none",

        bf16=True,

        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        remove_unused_columns=True,

        logging_dir=f"saved_models/adapters/{adapter_name}/logs"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    start = time()
    result = trainer.train()
    duration_s = time() - start

    model.save_pretrained(f"saved_models/adapters/{adapter_name}")

    # Return metrics for tracking
    return {
        "train_loss": round(result.training_loss, 4),
        "train_runtime_s": round(duration_s, 1),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    }


def traning(config_path):
    config = read_params(config_path)
    training_cfg = config["training"]
    base_model = training_cfg["base_model"]
    task_1_train_path = config["data"]["task_1_train_path"]
    task_1_val_path = config["data"]["task_1_val_path"]
    task_2_train_path = config["data"]["task_2_train_path"]
    task_2_val_path = config["data"]["task_2_val_path"]
    task_1_target_modules = training_cfg["task_1_target_modules"].split(",")
    task_2_target_modules = training_cfg["task_2_target_modules"].split(",")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=None,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    model = model.to("cuda")

    task_1_train_dataset = load_from_disk(task_1_train_path)
    task_1_val_dataset = load_from_disk(task_1_val_path)
    task_2_train_dataset = load_from_disk(task_2_train_path)
    task_2_val_dataset = load_from_disk(task_2_val_path)

    # Train Task 1
    task_1_metrics = train_function(
        model=model,
        tokenizer=tokenizer,
        train_dataset=task_1_train_dataset,
        eval_dataset=task_1_val_dataset,
        target_modules=task_1_target_modules,
        adapter_name="task_1",
        config=config
    )

    # Train Task 2 — reload base model since Task 1 modified it
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=None,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    model = model.to("cuda")

    task_2_metrics = train_function(
        model=model,
        tokenizer=tokenizer,
        train_dataset=task_2_train_dataset,
        eval_dataset=task_2_val_dataset,
        target_modules=task_2_target_modules,
        adapter_name="task_2",
        config=config
    )

    # Save completion marker
    with open("training_completion_time.txt", "w") as f:
        f.write(str(time()))

    # Save experiment metadata
    experiment = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "base_model": base_model,
        "hyperparameters": {
            "learning_rate": float(training_cfg["learning_rate"]),
            "num_epochs": int(training_cfg["num_epochs"]),
            "batch_size": int(training_cfg["batch_size"]),
            "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
            "lora_r": int(training_cfg["lora_r"]),
            "lora_alpha": int(training_cfg["lora_alpha"]),
            "lora_dropout": float(training_cfg["lora_dropout"]),
        },
        "task_1": {
            "target_modules": task_1_target_modules,
            **task_1_metrics,
        },
        "task_2": {
            "target_modules": task_2_target_modules,
            **task_2_metrics,
        },
    }

    with open("training_metrics.json", "w") as f:
        json.dump(experiment, f, indent=2)

    print(f"\nExperiment tracked -> training_metrics.json")
    print(f"  Task 1 loss: {task_1_metrics['train_loss']}  ({task_1_metrics['train_runtime_s']}s)")
    print(f"  Task 2 loss: {task_2_metrics['train_loss']}  ({task_2_metrics['train_runtime_s']}s)")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    traning(config_path=parsed_args.config)
