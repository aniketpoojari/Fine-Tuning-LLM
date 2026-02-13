import datetime
import os

import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi

# CONFIGURATION
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_REPO = "aniketp2009gmail/phi3-bilora-code-review"
FEEDBACK_REPO = "aniketp2009gmail/bilora-user-feedback"

st.set_page_config(page_title="BiLoRA Code Assistant", page_icon="🧠")

# Load the base model and adapters from Hub
@st.cache_resource(show_spinner="Loading model (this takes ~2 mins on CPU)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        attn_implementation="eager"
    )

    model = PeftModel.from_pretrained(model, ADAPTER_REPO, subfolder="task_1", adapter_name="task_1")
    model.load_adapter(ADAPTER_REPO, subfolder="task_2", adapter_name="task_2")
    model.set_adapter("task_1")
    model.eval()
    return model, tokenizer


def log_feedback(prompt, response, task, rating):
    """Save feedback to a Hugging Face Dataset repo."""
    try:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            try:
                hf_token = st.secrets["HF_TOKEN"]
            except (KeyError, FileNotFoundError):
                pass

        if not hf_token:
            st.warning("HF_TOKEN not configured. Feedback not saved.")
            return

        new_data = {
            "timestamp": [str(datetime.datetime.now())],
            "task": [task],
            "prompt": [prompt],
            "response": [response],
            "rating": [rating]
        }
        df = pd.DataFrame(new_data)

        api = HfApi(token=hf_token)
        csv_data = df.to_csv(index=False, header=False)

        api.create_repo(repo_id=FEEDBACK_REPO, repo_type="dataset", exist_ok=True)

        filename = f"feedback_{datetime.date.today()}.csv"
        api.upload_file(
            path_or_fileobj=csv_data.encode(),
            path_in_repo=filename,
            repo_id=FEEDBACK_REPO,
            repo_type="dataset",
            run_as_future=True
        )
    except Exception as e:
        st.warning(f"Could not save feedback: {e}")


# UI
st.title("🧠 BiLoRA: Dual-Adapter Code Assistant")

try:
    model, tokenizer = load_model()

    task_option = st.sidebar.radio(
        "Select Task:",
        options=["task_1", "task_2"],
        format_func=lambda x: "Code Generation" if x == "task_1" else "Docstring Generation"
    )

    user_input = st.text_area("Input:", height=150)

    if st.button("Generate"):
        if user_input.strip():
            with st.spinner("Generating..."):
                model.set_adapter(task_option)
                prefix = "Generate code: " if task_option == "task_1" else "Generate docstring: "
                suffix = "\nCode: " if task_option == "task_1" else "\nDocstring: "

                full_prompt = f"{prefix}{user_input}{suffix}"
                inputs = tokenizer(full_prompt, return_tensors="pt")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

                input_length = inputs['input_ids'].shape[1]
                output_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

                st.session_state.last_prompt = user_input
                st.session_state.last_response = output_text
                st.session_state.last_task = task_option
                st.session_state.generated = True

    if st.session_state.get("generated"):
        st.subheader("Result:")
        st.code(st.session_state.last_response, language="python")

        st.write("---")
        st.write("Was this helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                log_feedback(st.session_state.last_prompt, st.session_state.last_response, st.session_state.last_task, 1)
                st.success("Thanks for the feedback!")
        with col2:
            if st.button("👎 No"):
                log_feedback(st.session_state.last_prompt, st.session_state.last_response, st.session_state.last_task, 0)
                st.info("Feedback recorded.")

except Exception as e:
    st.error(f"Error loading model: {e}")
