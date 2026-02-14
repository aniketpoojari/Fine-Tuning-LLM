# BiLoRA: Dual Adapter Fine-Tuning for Code Generation & Docstring Generation

Multi-task fine-tuning system using LoRA adapters on Microsoft Phi-3-mini-4k-instruct. Two specialized adapters — one for code generation, one for docstring generation — share the same base model and can be switched at inference time without reloading.

## Features

- **Dual LoRA Adapters**: Task-specific adapters targeting different model layers (attention vs MLP)
- **DVC Pipeline**: Reproducible end-to-end pipeline from data download to adapter deployment
- **Experiment Tracking**: Hyperparameters, training loss, git hash, and duration logged per run
- **Automated Benchmarking**: Quality gate that blocks deployment if metrics regress
- **CI/CD**: GitHub Actions smoke test + auto-deploy to HF Spaces
- **Human-in-the-Loop Feedback**: User ratings collected in production, downloadable for retraining

## Hugging Face Hub

- **Model + Adapters**: [aniketp2009gmail/phi3-bilora-code-review](https://huggingface.co/aniketp2009gmail/phi3-bilora-code-review)
- **Live Demo**: [aniketp2009gmail/phi3-bilora-assistant](https://huggingface.co/spaces/aniketp2009gmail/phi3-bilora-assistant)
- **User Feedback Dataset**: [aniketp2009gmail/bilora-user-feedback](https://huggingface.co/datasets/aniketp2009gmail/bilora-user-feedback)

## Benchmark Results

Evaluated on 20 samples (10 code generation, 10 docstring generation) with Groq LLM-as-judge.

| Metric | BiLoRA (ours) | Phi-3 Base | Groq LLaMA-3.3-70B |
|--------|---------------|------------|---------------------|
| Code Gen Pass Rate | **94.2%** | 70.0% | 100.0% |
| Code Gen Quality (1-5) | **3.7** | 3.6 | 4.4 |
| Docstring BLEU | 0.026 | 0.054 | **0.126** |
| Docstring Quality (1-5) | 2.5 | 4.0 | **4.2** |
| Avg Latency | 33,499ms | 24,561ms | **434ms** |

## Pipeline

```
data-download → data-process → data-split → train → benchmark → push-adapters
                                               │          │
                                       training_metrics   results.json
                                       .json              → baseline.json
                                                            (auto-promoted)

git push → GitHub Actions smoke test → deploy to HF Spaces
                                              │
                                        user feedback → HF Dataset
                                              │
                                   fetch_feedback.py → training pairs
```

## Quickstart

### Prerequisites

```bash
pip install -r requirements.txt
```

### Set API Keys

```bash
export GROQ_API_KEY=gsk_...    # For LLM-as-judge evaluation
export HF_TOKEN=hf_...         # For pushing adapters and feedback
```

### Run Full Pipeline

```bash
dvc repro
```

This runs: download data → preprocess → split → train adapters → benchmark (quality gate) → push adapters to HF Hub.

### Run Individual Stages

```bash
dvc repro data-download
dvc repro data-process
dvc repro data-split
dvc repro train
dvc repro benchmark       # Fails if metrics regress — blocks push
dvc repro push-adapters   # Only runs after benchmark passes
```

### Experiment with Hyperparameters

```bash
# Edit params.yaml (e.g. change learning_rate, lora_r)
dvc repro

# See what changed
dvc params diff
dvc metrics diff
```

### Run Benchmark Standalone

```bash
# Full evaluation (all models + Groq judge)
python benchmarking/evaluate.py --groq-api-key "$GROQ_API_KEY"

# Quick smoke test
python benchmarking/evaluate.py --max-samples 1 --only-bilora

# Verbose (see raw model outputs)
python benchmarking/evaluate.py --groq-api-key "$GROQ_API_KEY" --verbose
```

### Deploy

```bash
# Commit and push — CI handles the rest
git add .
git commit -m "Experiment: increased lora_r to 8"
git push
# GitHub Actions: smoke test → deploy to HF Spaces
```

### Test HF Spaces App Locally

```bash
streamlit run hf_space/app.py
```

### Fetch User Feedback for Retraining

```bash
python scripts/fetch_feedback.py
# Outputs: data/feedback/all_feedback.json + training_pairs.json
```

## Project Structure

```
├── src/
│   ├── get_data.py            # Download MBPP + CodeXGLUE datasets
│   ├── preprocess_data.py     # Tokenize and format for training
│   ├── split_data.py          # Train/val split
│   ├── training.py            # Dual adapter LoRA training + experiment tracking
│   ├── push_adapters.py       # Upload adapters to HF Hub
│   └── common.py              # Config reader
├── benchmarking/
│   ├── evaluate.py            # Full benchmark suite (BiLoRA vs Base vs Groq)
│   ├── eval_dataset.json      # 20-sample test set
│   ├── results.json           # Latest evaluation results (DVC metric)
│   └── baseline.json          # Auto-updated regression baseline
├── scripts/
│   ├── compare_metrics.py     # Regression check + baseline promotion
│   └── fetch_feedback.py      # Download user feedback from HF Dataset
├── hf_space/
│   ├── app.py                 # Streamlit app with adapter switching + feedback
│   ├── requirements.txt       # Space dependencies
│   └── README.md              # HF Spaces metadata
├── .github/workflows/
│   └── deploy.yml             # CI: smoke test → deploy to HF Spaces
├── params.yaml                # All hyperparameters (DVC-tracked)
├── dvc.yaml                   # Pipeline definition
└── requirements.txt           # Project dependencies
```

## Implementation Details

- **Base Model**: Microsoft Phi-3-mini-4k-instruct (3.8B params)
- **Quantization**: 4-bit NF4 (bitsandbytes) for training, float16 for inference
- **Task 1 Adapter**: Targets attention layers (`qkv_proj`, `o_proj`) — trained on MBPP
- **Task 2 Adapter**: Targets MLP layers (`gate_up_proj`, `down_proj`) — trained on CodeXGLUE
- **LoRA Config**: r=4, alpha=8, dropout=0.1
- **Evaluation**: Functional test cases + BLEU + Groq LLM-as-judge (1-5 quality scale)
