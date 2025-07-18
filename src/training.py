import argparse
from src.common import read_params
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import torch
from time import time

def train_function(model, tokenizer, train_dataset, eval_dataset, target_modules, adapter_name):

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Add LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=f"saved_models/adapters/{adapter_name}",

        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=1,

        logging_steps=20,
        save_steps=100,
        save_total_limit=1,

        eval_strategy="steps",   # run evaluation during training
        eval_steps=100,

        load_best_model_at_end=True,  # restore best model by eval loss
        metric_for_best_model="loss",
        greater_is_better=False,

        report_to="none",  # disable wandb/logging unless used

        bf16=True,  # use bf16 if your GPU supports it (e.g., A100, 3090)
        # fp16=True if not bf16-compatible

        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        remove_unused_columns=True,

        logging_dir=f"saved_models/adapters/{adapter_name}/logs"
    )

    # Custom data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer # Changed from tokenizer to processing_class
    )

    trainer.train()
    model.save_pretrained(f"saved_models/adapters/{adapter_name}")

def traning(config_path):
    config = read_params(config_path)
    base_model = config["training"]["base_model"]
    task_1_train_path = config["data"]["task_1_train_path"]
    task_1_val_path = config["data"]["task_1_val_path"]
    task_2_train_path = config["data"]["task_2_train_path"]
    task_2_val_path = config["data"]["task_2_val_path"]
    task_1_target_modules = config["training"]["task_1_target_modules"].split(",")
    task_2_target_modules = config["training"]["task_2_target_modules"].split(",")

    # INITIALIZE TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # QUANTIZATION CONFIG
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

    # Load datasets from disk
    task_1_train_dataset = load_from_disk(task_1_train_path)
    task_1_val_dataset = load_from_disk(task_1_val_path)
    task_2_train_dataset = load_from_disk(task_2_train_path)
    task_2_val_dataset = load_from_disk(task_2_val_path)

    # Train Task 1
    train_function(
        model=model,
        tokenizer=tokenizer,
        train_dataset=task_1_train_dataset,
        eval_dataset=task_1_val_dataset,
        target_modules=task_1_target_modules,
        adapter_name="task_1"
    )


    # Train Task 2 - Need to reload base model since it was modified by Task 1 training
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=None,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    model = model.to("cuda")

    train_function(
        model=model,
        tokenizer=tokenizer,
        train_dataset=task_2_train_dataset,
        eval_dataset=task_2_val_dataset,
        target_modules=task_2_target_modules,
        adapter_name="task_2"
    )

    # save the time of completion in a text file
    with open("training_completion_time.txt", "w") as f:
        f.write(str(time()))



if __name__ == "__main__":
    # PARSE CONFIG FILE ARGUMENT
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    
    # START DOWNLOADING
    traning(config_path=parsed_args.config)