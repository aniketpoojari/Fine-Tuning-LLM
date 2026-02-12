# 🧠 BiLoRA: Dual Adapter Fine-Tuning for Code Generation & Docstring Generation

This project implements a multi-task fine-tuning system for Large Language Models using LoRA (Low-Rank Adaptation) adapters. The system can dynamically switch between different specialized adapters for code generation and docstring generation tasks, built on top of Microsoft's Phi-3-mini-4k-instruct model.

## 🚀 Features

- **Multi-Task Fine-Tuning**: Supports two specialized tasks:
  - **Code Generation**: Generates Python code from natural language descriptions (Fine-tuned on MBPP)
  - **Docstring Generation**: Creates documentation strings for Python code (Fine-tuned on CodeXGLUE)
- **Dynamic Adapter Switching**: Real-time switching between task-specific adapters without model reloading.
- **Efficient Training**: Uses 4-bit quantization for memory-efficient training.
- **Streamlit Interface**: User-friendly web interface for testing and switching adapters.
- **DVC Pipeline**: Complete data versioning and experiment tracking pipeline.
- **Hugging Face Hub**: Models and datasets are published for easy access.

## 📦 Hugging Face Hub

- **Model**: [aniketp2009gmail/phi3-bilora-code-review](https://huggingface.co/aniketp2009gmail/phi3-bilora-code-review)
- **Evaluation Dataset**: [aniketp2009gmail/code-review-benchmark](https://huggingface.co/datasets/aniketp2009gmail/code-review-benchmark)

## 📌 Benchmark Results

Evaluation performed on a custom benchmark of 20 samples.

| Model | Bug Detection (Pass@1) | Localization (BLEU) | Fix Quality (1-5) | Latency (avg) |
|-------|------------------------|---------------------|-------------------|---------------|
| **BiLoRA (mine)** | **94.17%** | **0.0259** | **3.7/5** | **33499ms** |
| Phi-3 base | 70.0% | 0.0536 | 3.6/5 | 24561ms |
| GPT-4 (Groq) | 100.0% | 0.1255 | 4.4/5 | 433ms |

*Note: Bug Detection is proxied by Code Generation Pass Rate. Localization is proxied by Docstring BLEU score.*

## 🔧 Quickstart

### Prerequisites
```bash
pip install -r requirements.txt
```

### Setup and Training

1. **Configure Parameters**
   Edit `params.yaml` to customize datasets, model settings, and training parameters.

2. **Run the Complete Pipeline**
   ```bash
   # Download and preprocess data
   dvc repro data-download
   dvc repro data-process
   dvc repro data-split
   
   # Train adapters
   dvc repro train
   ```

3. **Launch the Demo Interface**
   ```bash
   cd saved_models
   streamlit run app.py
   ```

## 🛠️ Implementation Details

- **Base Model**: Microsoft Phi-3-mini-4k-instruct
- **Quantization**: 4-bit (bitsandbytes)
- **Adapters**: PEFT LoRA
- **Training Strategy**: Individual adapter training per task with shared base model weights during inference.
