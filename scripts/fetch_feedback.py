"""
Download user feedback from HF Dataset and format for future retraining.

Usage:
    python scripts/fetch_feedback.py
    HF_TOKEN=hf_... python scripts/fetch_feedback.py
"""

import csv
import io
import json
import os
from datetime import datetime

from huggingface_hub import HfApi, list_repo_files

FEEDBACK_REPO = "aniketp2009gmail/bilora-user-feedback"
OUTPUT_DIR = "data/feedback"


def fetch_feedback():
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # List all feedback CSV files in the dataset repo
    try:
        files = list_repo_files(FEEDBACK_REPO, repo_type="dataset", token=token)
    except Exception as e:
        print(f"Could not access {FEEDBACK_REPO}: {e}")
        return

    csv_files = [f for f in files if f.startswith("feedback_") and f.endswith(".csv")]

    if not csv_files:
        print("No feedback files found.")
        return

    print(f"Found {len(csv_files)} feedback file(s)")

    # Download and parse all feedback
    all_rows = []
    for filename in sorted(csv_files):
        path = api.hf_hub_download(
            repo_id=FEEDBACK_REPO,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    all_rows.append({
                        "timestamp": row["timestamp"],
                        "task": row["task"],
                        "prompt": row["prompt"],
                        "response": row["response"],
                        "rating": int(row["rating"]),
                    })
                except (KeyError, ValueError):
                    continue

    if not all_rows:
        print("No feedback entries found in files.")
        return

    # Summary
    total = len(all_rows)
    positive = sum(1 for r in all_rows if r["rating"] == 1)
    negative = total - positive
    task_1 = sum(1 for r in all_rows if r["task"] == "task_1")
    task_2 = total - task_1

    print(f"\n--- Feedback Summary ---")
    print(f"Total entries:  {total}")
    print(f"Positive (thumbs up):   {positive} ({positive/total:.0%})")
    print(f"Negative (thumbs down): {negative} ({negative/total:.0%})")
    print(f"Code generation (task_1):     {task_1}")
    print(f"Docstring generation (task_2): {task_2}")

    # Save raw feedback
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_path = os.path.join(OUTPUT_DIR, "all_feedback.json")
    with open(raw_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nRaw feedback saved: {raw_path}")

    # Format positive feedback as training pairs for future fine-tuning
    # Only thumbs-up responses are useful as training signal
    training_pairs = []
    for row in all_rows:
        if row["rating"] != 1:
            continue

        if row["task"] == "task_1":
            text = f"Generate code: {row['prompt']}\nCode: {row['response']}"
        else:
            text = f"Generate docstring: {row['prompt']}\nDocstring: {row['response']}"

        training_pairs.append({
            "task": row["task"],
            "text": text,
        })

    if training_pairs:
        pairs_path = os.path.join(OUTPUT_DIR, "training_pairs.json")
        with open(pairs_path, "w") as f:
            json.dump(training_pairs, f, indent=2)
        print(f"Training pairs (positive only): {pairs_path} ({len(training_pairs)} samples)")
    else:
        print("No positive feedback to create training pairs from.")


if __name__ == "__main__":
    fetch_feedback()
