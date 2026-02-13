import streamlit as st
import torch
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# CONFIGURATION
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_REPO = "aniketp2009gmail/phi3-bilora-code-review"

# Task configurations
TASK_FORMATS = {
    "task_1": {
        "function": "Code Generation",
        "prompt": "Generate code: ",
        "answer": "Code: "
    },
    "task_2": {
        "function": "Docstring Generation", 
        "prompt": "Generate docstring: ",
        "answer": "Docstring: "
    }
}

st.set_page_config(page_title="BiLoRA Code Assistant", page_icon="🧠")

# Load the base model and adapters from Hub
@st.cache_resource(show_spinner="Loading model (this takes ~2 mins on CPU)...")
def load_model():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model in float16 for CPU efficiency
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Load Task 1 adapter from subfolder
    model = PeftModel.from_pretrained(
        model, 
        ADAPTER_REPO, 
        subfolder="task_1",
        adapter_name="task_1"
    )
    
    # Load Task 2 adapter from subfolder
    model.load_adapter(ADAPTER_REPO, subfolder="task_2", adapter_name="task_2")
    
    model.set_adapter("task_1")
    return model, tokenizer

def format_prompt_for_task(user_input, task_name):
    if task_name in TASK_FORMATS:
        task_format = TASK_FORMATS[task_name]
        return f"{task_format['prompt']}{user_input}\n{task_format['answer']}"
    return user_input

# UI
st.title("🧠 BiLoRA: Dual-Adapter Code Assistant")
st.markdown("""
This app uses **BiLoRA** (Dual-Adapter LoRA) fine-tuned on top of **Phi-3 Mini**. 
It can switch instantly between Code Generation and Docstring Generation.
""")

try:
    model, tokenizer = load_model()

    # Sidebar for task selection
    st.sidebar.title("Settings")
    task_option = st.sidebar.radio(
        "Select Task:",
        options=["task_1", "task_2"],
        format_func=lambda x: TASK_FORMATS[x]["function"]
    )

    if st.sidebar.button("Switch Adapter"):
        model.set_adapter(task_option)
        st.sidebar.success(f"Switched to {TASK_FORMATS[task_option]['function']}")

    # Main interaction
    user_input = st.text_area(
        "Input:", 
        height=150, 
        placeholder="e.g., Write a function to sort a list of strings by their length." if task_option == "task_1" else "e.g., def add(a, b): return a + b"
    )

    if st.button("Generate"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Generating on CPU (may take 30-60s)..."):
                model.set_adapter(task_option)
                formatted_prompt = format_prompt_for_task(user_input.strip(), task_option)
                
                inputs = tokenizer(formatted_prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                st.subheader("Result:")
                st.code(output_text, language="python")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Note: Loading might fail if the Space runs out of memory (16GB limit).")
