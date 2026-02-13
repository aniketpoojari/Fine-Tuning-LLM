import streamlit as st
import torch
import warnings
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Suppress warnings
warnings.filterwarnings("ignore", message=".*seen_tokens.*")
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"


# CONFIGURATION
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTERS_DIR = "adapters"

# Task configurations with paths and functions
TASK_FORMATS = {
    "task_1": {
        "path": os.path.join(ADAPTERS_DIR, "task_1"),
        "function": "Code Generation",
        "prompt": "Generate code: ",
        "answer": "Code: "
    },
    "task_2": {
        "path": os.path.join(ADAPTERS_DIR, "task_2"),
        "function": "Docstring Generation", 
        "prompt": "Generate docstring: ",
        "answer": "Docstring: "
    }
}

# Load the base model and all available adapters
@st.cache_resource(show_spinner="Loading model and adapters...")
def load_model_with_adapters():
    # Load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        attn_implementation="eager",
        use_cache=False,
    )
    
    # Load all available adapters
    loaded_adapters = []
    first_adapter = True
    first_adapter_name = None
    
    for adapter_name, config in TASK_FORMATS.items():
        adapter_path = config["path"]
        if os.path.exists(adapter_path):
            try:
                if first_adapter:
                    # Convert base model to PEFT model with first adapter
                    model = PeftModel.from_pretrained(
                        model, 
                        adapter_path, 
                        adapter_name=adapter_name,
                        is_trainable=False
                    )
                    first_adapter = False
                    first_adapter_name = adapter_name
                else:
                    # Load additional adapters
                    model.load_adapter(adapter_path, adapter_name=adapter_name)
                
                loaded_adapters.append(adapter_name)
            except Exception as e:
                st.error(f"Failed to load {adapter_name}: {str(e)}")
    
    # Set the first adapter as active
    if first_adapter_name and hasattr(model, 'set_adapter'):
        model.set_adapter(first_adapter_name)
    
    return model, tokenizer, loaded_adapters, first_adapter_name

def format_prompt_for_task(user_input, task_name):
    """Format the user input according to the task's training format"""
    if task_name in TASK_FORMATS:
        task_format = TASK_FORMATS[task_name]
        formatted_prompt = f"{task_format['prompt']}{user_input}\n{task_format['answer']}"
        return formatted_prompt
    else:
        return user_input

def set_adapter(model, adapter_name):
    """Activate a specific adapter"""
    try:
        if hasattr(model, 'set_adapter'):
            model.set_adapter(adapter_name)
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Failed to activate adapter {adapter_name}: {str(e)}")
        return False

# Load model, tokenizer and get loaded adapters
model, tokenizer, loaded_adapters, first_adapter_name = load_model_with_adapters()

# Initialize session state
if "active_adapter" not in st.session_state:
    st.session_state.active_adapter = first_adapter_name
if "show_loading_info" not in st.session_state:
    st.session_state.show_loading_info = True
if "loading_info_time" not in st.session_state:
    st.session_state.loading_info_time = time.time()

# Title
st.title("🔧 Adapter-Switchable LLM Generator")

# Display current adapter status
if st.session_state.active_adapter:
    adapter_function = TASK_FORMATS[st.session_state.active_adapter]["function"]
    st.info(f"Active: **{st.session_state.active_adapter}** ({adapter_function})")
else:
    st.warning("No adapter active - please select an adapter")

# Adapter control buttons
if loaded_adapters:
    cols = st.columns(len(loaded_adapters))
    
    # Create buttons for each loaded adapter
    for i, adapter_name in enumerate(loaded_adapters):
        with cols[i]:
            adapter_function = TASK_FORMATS[adapter_name]["function"]
            if st.button(f"🔁 {adapter_function}", key=f"btn_{adapter_name}"):
                if set_adapter(model, adapter_name):
                    st.session_state.active_adapter = adapter_name
                    st.rerun()
                else:
                    st.error(f"❌ Failed to activate {adapter_name}")
else:
    st.error("❌ No adapters loaded - check your adapter files")

# User input
user_input = st.text_area("📝 Enter your input:", height=150, placeholder="Enter your request here...")

# Generate output
if st.button("🚀 Generate"):
    if not user_input.strip():
        st.warning("Please enter a prompt.")
    else:
        try:
            with st.spinner("Generating..."):
                # Format the prompt according to the active task
                formatted_prompt = format_prompt_for_task(user_input.strip(), st.session_state.active_adapter)
                
                # Tokenize input
                inputs = tokenizer(
                    formatted_prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=1000
                )
                
                # Move inputs to model device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate with optimal sampling parameters
                generation_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs.get("attention_mask"),
                    "max_new_tokens": 200,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "use_cache": False,
                }
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(**generation_kwargs)
                
                # Decode output
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Display results
                st.text_area("Output:", value=output_text, height=200, disabled=True)
                
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")