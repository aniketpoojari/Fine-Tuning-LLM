"""
Push trained adapters to Hugging Face Hub.

Usage:
    python src/push_adapters.py --config params.yaml
    HF_TOKEN=hf_... python src/push_adapters.py --config params.yaml
"""

import argparse
import os

from huggingface_hub import HfApi
from src.common import read_params


HUB_REPO = "aniketp2009gmail/phi3-bilora-code-review"


def push_adapters(config_path):
    config = read_params(config_path)
    base_model = config["training"]["base_model"]

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is required")

    api = HfApi(token=token)

    # Ensure repo exists
    api.create_repo(repo_id=HUB_REPO, exist_ok=True)

    # Also push the tokenizer from the base model so the Space can load it
    # from the adapter repo directly
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.push_to_hub(HUB_REPO, token=token)

    for adapter_name in ["task_1", "task_2"]:
        local_path = f"saved_models/adapters/{adapter_name}"
        if not os.path.exists(local_path):
            print(f"  SKIP: {local_path} not found")
            continue

        print(f"  Uploading {adapter_name} -> {HUB_REPO}/{adapter_name} ...")
        api.upload_folder(
            folder_path=local_path,
            path_in_repo=adapter_name,
            repo_id=HUB_REPO,
        )
        print(f"  Done: {adapter_name}")

    print(f"\nAdapters pushed to https://huggingface.co/{HUB_REPO}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    push_adapters(config_path=args.config)
