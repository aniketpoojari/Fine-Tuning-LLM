import json
import shutil
import sys
import os

def compare_metrics(new_file, baseline_file, threshold=0.02):
    if not os.path.exists(baseline_file):
        print("No baseline found. Skipping comparison.")
        return True

    with open(new_file) as f:
        new = json.load(f)["metrics"]["bilora"]
    with open(baseline_file) as f:
        baseline = json.load(f)["metrics"]["bilora"]

    # Metrics to check (higher is better)
    to_check = ["codegen_pass_rate", "codegen_quality (1-5)", "docgen_bleu"]

    failed = False
    for metric in to_check:
        old_val = baseline.get(metric, 0)
        new_val = new.get(metric, 0)

        # Skip metrics where the new value is 0 (e.g. quality scores
        # are 0 when Groq judge is unavailable or --only-bilora is used)
        if new_val == 0:
            print(f"Skipping {metric}: New={new_val} (not evaluated)")
            continue

        print(f"Checking {metric}: Baseline={old_val}, New={new_val}")

        if new_val < (old_val - threshold):
            print(f"REGRESSION DETECTED: {metric} dropped significantly!")
            failed = True

    return not failed

if __name__ == "__main__":
    new_results = sys.argv[1] if len(sys.argv) > 1 else "benchmarking/results.json"
    baseline = sys.argv[2] if len(sys.argv) > 2 else "benchmarking/baseline.json"

    if not compare_metrics(new_results, baseline):
        sys.exit(1)

    print("Quality check passed!")

    # Promote current results to new baseline
    if os.path.abspath(new_results) != os.path.abspath(baseline):
        shutil.copy2(new_results, baseline)
        print(f"Baseline updated: {baseline}")
