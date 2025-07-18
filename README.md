# 🧠 BiLoRA: Dual Adapter Fine-Tuning for Code Generation & Docstring Generation

This project implements a multi-task fine-tuning system for Large Language Models using LoRA (Low-Rank Adaptation) adapters. The system can dynamically switch between different specialized adapters for code generation and docstring generation tasks, built on top of Microsoft's Phi-3-mini-4k-instruct model[1].

## 🚀 Features

- **Multi-Task Fine-Tuning**: Supports two specialized tasks:
  - **Code Generation**: Generates Python code from natural language descriptions
  - **Docstring Generation**: Creates documentation strings for Python code
- **Dynamic Adapter Switching**: Real-time switching between task-specific adapters without model reloading
- **Efficient Training**: Uses 4-bit quantization with BitsAndBytesConfig for memory-efficient training
- **Streamlit Interface**: User-friendly web interface for testing and demonstration
- **DVC Pipeline**: Complete data versioning and experiment tracking pipeline
- **Memory Optimization**: Implements gradient checkpointing and optimized batch processing

## 🔧 Quickstart

### Prerequisites
```bash
pip install -r requirements.txt
```

### Setup and Training

1. **Configure Parameters**
   Edit `params.yaml` to customize datasets, model settings, and training parameters:
   ```yaml
   data:
     task_1_dataset_name: mbpp
     task_2_dataset_name: google/code_x_glue_ct_code_to_text
   training:
     base_model: microsoft/Phi-3-mini-4k-instruct
   ```

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

## 📌 Results

The system demonstrates successful multi-task learning with the following capabilities:

| Task | Function | Input Format | Output Format |
|------|----------|--------------|---------------|
| Task 1 | Code Generation | Natural language description | Python code |
| Task 2 | Docstring Generation | Python code | Documentation string |

**Key Performance Features:**
- **Memory Efficient**: 4-bit quantization reduces GPU memory requirements by ~75%
- **Fast Switching**: Adapter switching occurs in milliseconds without model reloading
- **Scalable Architecture**: Easy to add new tasks by training additional adapters
- **Production Ready**: Streamlit interface provides immediate deployment capability

The trained adapters are saved in `saved_models/adapters/` and can be independently loaded and switched during inference, making the system highly flexible for different coding assistance tasks.