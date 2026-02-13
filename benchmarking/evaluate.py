"""
Evaluation suite for BiLoRA fine-tuned model.

Benchmarks both trained tasks against Base Phi-3 and a Groq cloud model:
  - Task 1 (code generation):  adapter=task_1, prompt="Generate code: ...\\nCode: "
  - Task 2 (docstring generation): adapter=task_2, prompt="Generate docstring: ...\\nDocstring: "

Usage:
    python benchmarking/evaluate.py --groq-api-key <KEY>
    python benchmarking/evaluate.py                          # local only
    python benchmarking/evaluate.py --max-samples 4          # quick smoke test
    python benchmarking/evaluate.py --verbose                # print raw model outputs
"""

import copy
import json
import time
import argparse
import math
import os
import re
import sys
from collections import Counter

import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


VERBOSE = False


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

# Must match preprocess_data.py exactly
LOCAL_FORMATS = {
    "code_generation":     {"prompt": "Generate code: ", "answer": "Code: "},
    "docstring_generation": {"prompt": "Generate docstring: ", "answer": "Docstring: "},
}


def build_local_prompt(task, user_input):
    """Prompt for BiLoRA / base Phi-3 — matches training format exactly."""
    fmt = LOCAL_FORMATS[task]
    return fmt["prompt"] + user_input + "\n" + fmt["answer"]


def build_groq_prompt_codegen(task_description):
    """Chat-style prompt for Groq code generation."""
    return (
        "Write a Python function for the following task. "
        "Return ONLY the function definition. No explanation, no examples, "
        "no markdown fences.\n\n"
        f"Task: {task_description}"
    )


def build_groq_prompt_docstring(code):
    """Chat-style prompt for Groq docstring generation."""
    return (
        "Write a brief docstring for the following Python function. "
        "Return ONLY the docstring text, no quotes, no code, no markdown.\n\n"
        f"{code}"
    )


# ---------------------------------------------------------------------------
# Code extraction  (the previous version was too naive)
# ---------------------------------------------------------------------------

def extract_function(text):
    """
    Extract the first complete Python function from raw model output.

    Strategy:
      1. Strip markdown fences if present.
      2. Find the first 'def ' line.
      3. Collect all subsequent lines that are indented deeper (the function body).
      4. Stop at the first line that de-indents back to the def's level (or at EOF).
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:python)?\s*\n?", "", text)
    text = text.replace("```", "")

    lines = text.split("\n")

    # Find first 'def '
    start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def "):
            start = i
            break

    if start is None:
        return text.strip()

    # Determine base indentation of the def line
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    func_lines = [lines[start]]

    for i in range(start + 1, len(lines)):
        line = lines[i]

        # Blank line inside a function is fine
        if line.strip() == "":
            func_lines.append(line)
            continue

        # Check indentation
        current_stripped = line.lstrip()
        current_indent = len(line) - len(current_stripped)

        # Still inside the function body
        if current_indent > base_indent:
            func_lines.append(line)
        elif current_stripped.startswith("#") or current_stripped.startswith('"""') or current_stripped.startswith("'''"):
            # Comments at base indent might still be part of the "thought process" but we usually want to stop
            # unless it's a docstring immediately following 'def'.
            # For simplicity, if it's at base indent, we stop.
            break
        else:
            # Hit something at the same or lesser indent -> function ended
            break

    # Trim trailing blank lines
    while func_lines and func_lines[-1].strip() == "":
        func_lines.pop()

    result = "\n".join(func_lines)

    # Sanity check: does it compile?
    try:
        compile(result, "<eval>", "exec")
    except SyntaxError:
        # If the extracted code has a syntax error, try to return it anyway
        # but maybe it's because we included a trailing line that was bad.
        # If it's totally broken, run_test_cases will handle it.
        return result.strip()

    return result


def clean_docstring(text):
    """Clean raw model output into a plain docstring string."""
    text = text.strip()
    # Remove markdown fences
    text = re.sub(r"```(?:python)?\s*\n?", "", text)
    text = text.replace("```", "")
    # Remove surrounding quotes (model sometimes wraps in triple quotes)
    for q in ['"""', "'''", '"', "'"]:
        if text.startswith(q) and text.endswith(q):
            text = text[len(q):-len(q)]
    return text.strip()


# ---------------------------------------------------------------------------
# BLEU score (self-contained)
# ---------------------------------------------------------------------------

def tokenize_code(text):
    """Tokenize code/text by splitting on whitespace and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def compute_bleu(reference, hypothesis, max_n=4):
    """Sentence-level BLEU-4 with smoothing and better tokenization."""
    ref_tokens = tokenize_code(reference)
    hyp_tokens = tokenize_code(hypothesis)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1)
        )
        if not hyp_ngrams:
            # Smoothing for missing higher-order n-grams
            precisions.append(0.0)
            continue
        
        clipped = sum(
            min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items()
        )
        total = sum(hyp_ngrams.values())
        
        # Smoothing: if precision is 0, use a small floor value
        if clipped == 0:
            precisions.append(1 / (2 * total) if total > 0 else 0.0)
        else:
            precisions.append(clipped / total)

    if not any(precisions):
        return 0.0

    # Avoid log(0) if any precision is still 0
    precisions = [p if p > 0 else 1e-9 for p in precisions]

    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    log_avg = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# Functional correctness
# ---------------------------------------------------------------------------

def run_test_cases(generated_code, function_name, test_cases):
    """
    Execute generated code and test against expected outputs.
    Tries the expected function name first, then falls back to the first
    callable found (model may pick its own name).
    Returns (passed, total).
    """
    total = len(test_cases)

    # Execute once
    try:
        # Create a clean sandbox-like dict for execution
        local_ns = {}
        exec(generated_code, {"__builtins__": __builtins__}, local_ns)
    except Exception as e:
        if VERBOSE:
            print(f"\n    [EXEC FAIL] {e}")
        return 0, total

    # Find function: try expected name, then first callable
    func = local_ns.get(function_name)
    if func is None:
        # Fallback: look for ANY callable in the namespace that isn't a builtin
        for name, obj in local_ns.items():
            if callable(obj) and not isinstance(obj, type) and not name.startswith("__"):
                if VERBOSE:
                    print(f"\n    [NAME MISS] expected '{function_name}', found '{name}'")
                func = obj
                break

    if func is None:
        if VERBOSE:
            print(f"\n    [NO FUNC] no callable found in generated code")
        return 0, total

    passed = 0
    for i, tc in enumerate(test_cases):
        try:
            inp = copy.deepcopy(tc["input"])
            result = func(*inp)
            if result == tc["expected"]:
                passed += 1
            elif VERBOSE:
                print(f"\n    [TEST {i+1} FAIL] got {result!r}, expected {tc['expected']!r}")
        except Exception as e:
            if VERBOSE:
                print(f"\n    [TEST {i+1} ERROR] {e}")

    return passed, total


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_with_adapters(base_model_name, adapter_dir):
    """
    Load Phi-3 with both LoRA adapters.
    Tries local paths first, falls back to Hugging Face Hub.
    """
    device = get_device()
    use_bnb = device == "cuda"
    hub_repo = "aniketp2009gmail/phi3-bilora-code-review"

    bnb_config = None
    if use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    print(f"  Loading base model: {base_model_name} (device={device}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto" if use_bnb else None,
        torch_dtype=torch.bfloat16 if use_bnb else torch.float16,
        low_cpu_mem_usage=True,
        use_cache=True,
        trust_remote_code=False,
        attn_implementation="eager"
    )
    if not use_bnb:
        model = model.to(device)

    loaded = []
    first = True
    for adapter_name in ["task_1", "task_2"]:
        local_path = os.path.join(adapter_dir, adapter_name)
        
        # Determine source (Local vs Hub)
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "adapter_config.json")):
            load_path = local_path
            print(f"  Loading adapter '{adapter_name}' from LOCAL")
        else:
            load_path = hub_repo
            print(f"  Loading adapter '{adapter_name}' from HUB ({hub_repo})")

        try:
            if first:
                model = PeftModel.from_pretrained(
                    model, load_path, 
                    subfolder=adapter_name if load_path == hub_repo else None,
                    adapter_name=adapter_name, 
                    is_trainable=False
                )
                first = False
            else:
                model.load_adapter(
                    load_path, 
                    subfolder=adapter_name if load_path == hub_repo else None,
                    adapter_name=adapter_name
                )
            loaded.append(adapter_name)
            print(f"  Successfully loaded adapter: {adapter_name}")
        except Exception as e:
            print(f"  ERROR: Could not load adapter {adapter_name}: {e}")
            # If the first load fails, we don't want to convert to PeftModel or first remains True
            # We continue to the next one

    if loaded:
        model.set_adapter(loaded[0])

    model.eval()
    return model, tokenizer, loaded


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_local(model, tokenizer, prompt, max_new_tokens=256):
    """
    Generate from local model.
    Uses add_special_tokens=False to match training tokenization.
    Uses greedy decoding for deterministic evaluation.
    Returns (text, latency_ms, memory_mb).
    """
    device = next(model.parameters()).device

    # Match training: add_special_tokens=False
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                    # greedy — deterministic
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        memory_mb = psutil.Process().memory_info().rss / 1e6

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, latency_ms, memory_mb


def query_groq(client, prompt, model_name):
    """Call Groq chat API. Returns (text, latency_ms)."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,          # deterministic
        max_tokens=1024,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    return response.choices[0].message.content, latency_ms


# ---------------------------------------------------------------------------
# Groq-as-judge
# ---------------------------------------------------------------------------

CODE_JUDGE = """Rate the quality of generated Python code against a reference.

Task: {task_prompt}

Reference:
```python
{reference}
```

Generated:
```python
{generated}
```

1 = Wrong / does not define expected function
2 = Related but has bugs
3 = Mostly correct, fails edge cases
4 = Correct, minor style differences
5 = Functionally equivalent or better

Respond with ONLY a single integer (1-5):"""

DOCSTRING_JUDGE = """Rate a generated docstring against a reference.

Function:
```python
{code}
```

Reference docstring: {reference}
Generated docstring: {generated}

1 = Wrong or irrelevant
2 = Vaguely related, missing key details
3 = Describes function but incomplete
4 = Good, minor omissions
5 = Comprehensive and accurate

Respond with ONLY a single integer (1-5):"""


def judge_quality(client, groq_model, prompt_text):
    """Send judge prompt to Groq, return int 1-5 (0 on failure)."""
    try:
        text, _ = query_groq(client, prompt_text, groq_model)
        match = re.search(r"[1-5]", text.strip())
        return int(match.group()) if match else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def _run_local(model, tokenizer, prompt, adapter_name, loaded_adapters, use_adapter):
    """Run a single local inference with adapter toggling."""
    # Check if this is a PEFT model and has active adapters
    is_peft = hasattr(model, "set_adapter") and hasattr(model, "peft_config") and len(model.peft_config) > 0

    if is_peft:
        try:
            if use_adapter and adapter_name in loaded_adapters:
                if hasattr(model, "enable_adapters"):
                    model.enable_adapters()
                model.set_adapter(adapter_name)
            else:
                if hasattr(model, "disable_adapters"):
                    model.disable_adapters()
        except Exception as e:
            if VERBOSE:
                print(f"  WARNING: Adapter toggling failed: {e}")
            pass

    try:
        raw, lat, mem = generate_local(model, tokenizer, prompt)
    except Exception as e:
        raw, lat, mem = f"ERROR: {e}", 0.0, 0.0

    # Always restore adapter state if we were testing base
    if is_peft and not use_adapter:
        try:
            if hasattr(model, "enable_adapters"):
                model.enable_adapters()
            if loaded_adapters:
                model.set_adapter(loaded_adapters[0])
        except:
            pass

    return raw, lat, mem


def evaluate_codegen(model, tokenizer, groq_client, groq_model, sample, loaded_adapters, only_bilora=False):
    """Evaluate a single code-generation sample across models."""
    local_prompt = build_local_prompt("code_generation", sample["prompt"])
    fname = sample["function_name"]
    test_cases = sample["test_cases"]
    ref_code = sample["reference_code"]
    results = {}

    # --- BiLoRA ---
    raw, lat, mem = _run_local(
        model, tokenizer, local_prompt, "task_1", loaded_adapters, use_adapter=True
    )
    code = extract_function(raw)
    if VERBOSE:
        print(f"\n    [BiLoRA raw] {raw[:300]}...")
        print(f"    [BiLoRA extracted] {code[:200]}")
    passed, total = run_test_cases(code, fname, test_cases)
    results["bilora"] = {
        "generated_code": code,
        "test_passed": passed, "test_total": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "bleu": round(compute_bleu(ref_code, code), 4),
        "quality_score": 0,
        "latency_ms": round(lat, 1), "memory_mb": round(mem, 1),
        "raw_response": raw,
    }

    if only_bilora:
        results["base_phi3"] = _skip_result("Skipped (--only-bilora)")
        results["groq"] = _skip_result("Skipped (--only-bilora)")
        return results

    # --- Base Phi-3 ---
    raw, lat, mem = _run_local(
        model, tokenizer, local_prompt, "task_1", loaded_adapters, use_adapter=False
    )
    code = extract_function(raw)
    if VERBOSE:
        print(f"\n    [Base raw] {raw[:300]}...")
        print(f"    [Base extracted] {code[:200]}")
    passed, total = run_test_cases(code, fname, test_cases)
    results["base_phi3"] = {
        "generated_code": code,
        "test_passed": passed, "test_total": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "bleu": round(compute_bleu(ref_code, code), 4),
        "quality_score": 0,
        "latency_ms": round(lat, 1), "memory_mb": round(mem, 1),
        "raw_response": raw,
    }

    # --- Groq ---
    if groq_client:
        groq_prompt = build_groq_prompt_codegen(sample["prompt"])
        try:
            raw, lat = query_groq(groq_client, groq_prompt, groq_model)
        except Exception as e:
            raw, lat = f"ERROR: {e}", 0.0
        code = extract_function(raw)
        if VERBOSE:
            print(f"\n    [Groq raw] {raw[:300]}...")
        passed, total = run_test_cases(code, fname, test_cases)
        results["groq"] = {
            "generated_code": code,
            "test_passed": passed, "test_total": total,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "bleu": round(compute_bleu(ref_code, code), 4),
            "quality_score": 0,
            "latency_ms": round(lat, 1), "memory_mb": 0.0,
            "raw_response": raw,
        }
    else:
        results["groq"] = _skip_result()

    # --- Judge ---
    if groq_client:
        for key in ["bilora", "base_phi3", "groq"]:
            results[key]["quality_score"] = judge_quality(
                groq_client, groq_model,
                CODE_JUDGE.format(
                    task_prompt=sample["prompt"],
                    reference=ref_code,
                    generated=results[key]["generated_code"],
                ),
            )

    return results


def evaluate_docstring(model, tokenizer, groq_client, groq_model, sample, loaded_adapters, only_bilora=False):
    """Evaluate a single docstring-generation sample across models."""
    local_prompt = build_local_prompt("docstring_generation", sample["code"])
    ref = sample["reference_docstring"]
    results = {}

    # --- BiLoRA ---
    raw, lat, mem = _run_local(
        model, tokenizer, local_prompt, "task_2", loaded_adapters, use_adapter=True
    )
    docstring = clean_docstring(raw)
    if VERBOSE:
        print(f"\n    [BiLoRA raw] {raw[:300]}...")
        print(f"    [BiLoRA cleaned] {docstring[:200]}")
    results["bilora"] = {
        "generated_docstring": docstring,
        "bleu": round(compute_bleu(ref, docstring), 4),
        "quality_score": 0,
        "latency_ms": round(lat, 1), "memory_mb": round(mem, 1),
        "raw_response": raw,
    }

    if only_bilora:
        results["base_phi3"] = _skip_result("Skipped (--only-bilora)")
        results["groq"] = _skip_result("Skipped (--only-bilora)")
        return results

    # --- Base Phi-3 ---
    raw, lat, mem = _run_local(
        model, tokenizer, local_prompt, "task_2", loaded_adapters, use_adapter=False
    )
    docstring = clean_docstring(raw)
    if VERBOSE:
        print(f"\n    [Base raw] {raw[:300]}...")
        print(f"    [Base cleaned] {docstring[:200]}")
    results["base_phi3"] = {
        "generated_docstring": docstring,
        "bleu": round(compute_bleu(ref, docstring), 4),
        "quality_score": 0,
        "latency_ms": round(lat, 1), "memory_mb": round(mem, 1),
        "raw_response": raw,
    }

    # --- Groq ---
    if groq_client:
        groq_prompt = build_groq_prompt_docstring(sample["code"])
        try:
            raw, lat = query_groq(groq_client, groq_prompt, groq_model)
        except Exception as e:
            raw, lat = f"ERROR: {e}", 0.0
        docstring = clean_docstring(raw)
        if VERBOSE:
            print(f"\n    [Groq raw] {raw[:300]}...")
        results["groq"] = {
            "generated_docstring": docstring,
            "bleu": round(compute_bleu(ref, docstring), 4),
            "quality_score": 0,
            "latency_ms": round(lat, 1), "memory_mb": 0.0,
            "raw_response": raw,
        }
    else:
        results["groq"] = _skip_result()

    # --- Judge ---
    if groq_client:
        for key in ["bilora", "base_phi3", "groq"]:
            gen = results[key].get("generated_docstring", "")
            results[key]["quality_score"] = judge_quality(
                groq_client, groq_model,
                DOCSTRING_JUDGE.format(code=sample["code"], reference=ref, generated=gen),
            )

    return results


def _skip_result(reason="SKIPPED (no Groq API key)"):
    return {
        "generated_code": "", "generated_docstring": "",
        "test_passed": 0, "test_total": 0, "pass_rate": 0.0,
        "bleu": 0.0, "quality_score": 0,
        "latency_ms": 0.0, "memory_mb": 0.0,
        "raw_response": reason,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_results):
    codegen = [r for r in all_results if r["task"] == "code_generation"]
    docgen  = [r for r in all_results if r["task"] == "docstring_generation"]

    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    metrics = {}
    for mk in ["bilora", "base_phi3", "groq"]:
        cg_pass = [r[mk]["pass_rate"]     for r in codegen]
        cg_bleu = [r[mk]["bleu"]          for r in codegen]
        cg_qual = [r[mk]["quality_score"] for r in codegen]
        cg_lat  = [r[mk]["latency_ms"]    for r in codegen]
        cg_mem  = [r[mk]["memory_mb"]     for r in codegen if r[mk]["memory_mb"] > 0]

        dg_bleu = [r[mk]["bleu"]          for r in docgen]
        dg_qual = [r[mk]["quality_score"] for r in docgen]
        dg_lat  = [r[mk]["latency_ms"]    for r in docgen]
        dg_mem  = [r[mk]["memory_mb"]     for r in docgen if r[mk]["memory_mb"] > 0]

        all_lat = cg_lat + dg_lat
        all_mem = cg_mem + dg_mem

        metrics[mk] = {
            "codegen_pass_rate":     avg(cg_pass),
            "codegen_bleu":          avg(cg_bleu),
            "codegen_quality (1-5)": avg(cg_qual),
            "docgen_bleu":           avg(dg_bleu),
            "docgen_quality (1-5)":  avg(dg_qual),
            "avg_latency_ms":        round(avg(all_lat), 1),
            "avg_memory_mb":         round(avg(all_mem), 1),
        }

    return metrics


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_table(metrics):
    headers = ["Metric", "BiLoRA", "Base Phi-3", "Groq"]
    rows = []
    for key in list(next(iter(metrics.values())).keys()):
        rows.append([
            key,
            str(metrics["bilora"][key]),
            str(metrics["base_phi3"][key]),
            str(metrics["groq"][key]),
        ])

    col_widths = [max(len(r[i]) for r in [headers] + rows) for i in range(4)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-" * (sum(col_widths) + 6)

    print("\n" + "=" * len(sep))
    print("  BENCHMARK RESULTS")
    print("=" * len(sep))
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print("=" * len(sep))


def print_per_sample(all_results):
    codegen = [r for r in all_results if r["task"] == "code_generation"]
    docgen  = [r for r in all_results if r["task"] == "docstring_generation"]

    if codegen:
        print("\n--- Code Generation: Test Pass Rates ---")
        print(f"{'ID':>3}  {'Function':<20} {'BiLoRA':<14} {'Base':<14} {'Groq':<14}")
        print("-" * 68)
        for r in codegen:
            def fmt(mk):
                p = r[mk]["test_passed"]
                t = r[mk]["test_total"]
                rate = r[mk]["pass_rate"]
                return f"{p}/{t} ({rate:.0%})"
            print(
                f"{r['id']:>3}  {r['function_name']:<20} "
                f"{fmt('bilora'):<14} {fmt('base_phi3'):<14} {fmt('groq'):<14}"
            )

    if docgen:
        print("\n--- Docstring Generation: BLEU Scores ---")
        print(f"{'ID':>3}  {'Function':<28} {'BiLoRA':<10} {'Base':<10} {'Groq':<10}")
        print("-" * 64)
        for r in docgen:
            match = re.search(r"def (\w+)", r.get("code", ""))
            fname = match.group(1) if match else f"sample_{r['id']}"
            print(
                f"{r['id']:>3}  {fname:<28} "
                f"{r['bilora']['bleu']:<10.4f} "
                f"{r['base_phi3']['bleu']:<10.4f} "
                f"{r['groq']['bleu']:<10.4f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global VERBOSE

    parser = argparse.ArgumentParser(description="BiLoRA benchmark suite")
    parser.add_argument("--dataset",     default="benchmarking/eval_dataset.json")
    parser.add_argument("--base-model",  default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--adapter-dir", default="saved_models/adapters")
    parser.add_argument("--groq-api-key", default=None, help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--groq-model",  default="llama-3.3-70b-versatile")
    parser.add_argument("--output",      default="benchmarking/results.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--only-bilora", action="store_true", help="Only evaluate BiLoRA, skip benchmarking models")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print raw model outputs for debugging")
    args = parser.parse_args()
    VERBOSE = args.verbose

    # ── Load dataset ──────────────────────────────────────────────────
    print(f"\n[1/4] Loading dataset: {args.dataset}")
    with open(args.dataset) as f:
        dataset = json.load(f)
    
    if args.max_samples:
        # Per-task sampling: pick N samples from EACH task
        cg_all = [s for s in dataset if s["task"] == "code_generation"]
        dg_all = [s for s in dataset if s["task"] == "docstring_generation"]
        
        dataset = cg_all[:args.max_samples] + dg_all[:args.max_samples]
        print(f"  Using {len(dataset)} samples (up to {args.max_samples} from each task)")

    cg = [s for s in dataset if s["task"] == "code_generation"]
    dg = [s for s in dataset if s["task"] == "docstring_generation"]
    print(f"  {len(dataset)} samples  ({len(cg)} code gen, {len(dg)} docstring gen)")

    # ── Load model ────────────────────────────────────────────────────
    print(f"\n[2/4] Loading model + adapters")
    if not PEFT_AVAILABLE:
        sys.exit("  ERROR: 'peft' not installed.  pip install peft")
    model, tokenizer, loaded_adapters = load_model_with_adapters(
        args.base_model, args.adapter_dir
    )
    print(f"  Adapters loaded: {loaded_adapters or 'NONE'}")

    # ── Init Groq ─────────────────────────────────────────────────────
    groq_client = None
    groq_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")
    if groq_key and GROQ_AVAILABLE:
        groq_client = Groq(api_key=groq_key)
        print(f"  Groq ready ({args.groq_model})")
    else:
        reason = "no API key" if not groq_key else "'groq' not installed"
        print(f"  Groq skipped ({reason})")

    # ── Evaluate ──────────────────────────────────────────────────────
    print(f"\n[3/4] Running benchmark ...")
    all_results = []

    for idx, sample in enumerate(dataset):
        task = sample["task"]
        label = sample.get("function_name") or f"sample_{sample['id']}"
        print(f"  [{idx+1}/{len(dataset)}] {task}: {label} ...", end="" if not VERBOSE else "\n", flush=True)

        if task == "code_generation":
            res = evaluate_codegen(
                model, tokenizer, groq_client, args.groq_model, sample, loaded_adapters, only_bilora=args.only_bilora
            )
        else:
            res = evaluate_docstring(
                model, tokenizer, groq_client, args.groq_model, sample, loaded_adapters, only_bilora=args.only_bilora
            )

        entry = {
            "id": sample["id"],
            "task": task,
            "function_name": sample.get("function_name", ""),
            "code": sample.get("code", ""),
            **res,
        }
        all_results.append(entry)
        if not VERBOSE:
            # One-line summary
            if task == "code_generation":
                b = res["bilora"]["pass_rate"]
                p = res["base_phi3"]["pass_rate"]
                g = res["groq"]["pass_rate"]
                print(f" pass: BiLoRA={b:.0%}  Base={p:.0%}  Groq={g:.0%}")
            else:
                b = res["bilora"]["bleu"]
                p = res["base_phi3"]["bleu"]
                g = res["groq"]["bleu"]
                print(f" BLEU: BiLoRA={b:.4f}  Base={p:.4f}  Groq={g:.4f}")

    # ── Metrics ───────────────────────────────────────────────────────
    print(f"\n[4/4] Computing metrics ...")
    metrics = compute_metrics(all_results)

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "config": {
            "base_model": args.base_model,
            "adapters": loaded_adapters,
            "groq_model": args.groq_model if groq_client else None,
            "num_samples": len(dataset),
        },
        "metrics": metrics,
        "results": all_results,
    }
    for r in output["results"]:
        for mk in ["bilora", "base_phi3", "groq"]:
            if mk in r:
                raw = r[mk].get("raw_response", "")
                r[mk]["raw_response"] = (raw[:500] + "...") if len(raw) > 500 else raw

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ── Display ───────────────────────────────────────────────────────
    print_per_sample(all_results)
    print_table(metrics)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
