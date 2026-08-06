"""
Aggregate IFEval + MMLU results across all Mistral-7B conditions into
a single CSV.

Run after mistral_eval.py completes:
    python aggregate_mistral_eval.py

Expects the directory structure produced by lm-evaluation-harness under
--output_path, i.e. each condition folder containing a results.json
(lm-eval nests it under a model-name subfolder by default — this script
searches recursively for results*.json to handle that).
"""

import csv
import json
import sys
from pathlib import Path

# Make config.py (at the repo root) importable regardless of where this
# script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import EVAL_OUTPUT_ROOT

OUTPUT_ROOT = EVAL_OUTPUT_ROOT / "mistral"
CSV_OUT = OUTPUT_ROOT / "ifeval_mmlu_summary.csv"

# The four standard IFEval metrics
IFEVAL_METRICS = [
    "prompt_level_strict_acc,none",
    "inst_level_strict_acc,none",
    "prompt_level_loose_acc,none",
    "inst_level_loose_acc,none",
]

# MMLU: lm-eval reports an overall "acc,none" under the mmlu group,
# plus per-subject scores under mmlu_<subject>. We pull the overall here;
# per-subject breakdowns are in the raw results.json if needed later.
MMLU_METRIC = "acc,none"


def parse_condition(folder_name: str):
    """mistral7b_r4_early -> (rank=4, placement=early); 'base' -> (None, 'base');
    'full_ft' -> (None, 'full_ft')"""
    if folder_name == "base":
        return None, "base"
    if folder_name == "full_ft":
        return None, "full_ft"
    parts = folder_name.replace("mistral7b_", "").split("_")
    rank = int(parts[0].lstrip("r"))
    placement = parts[1]
    return rank, placement


def find_results_json(condition_dir: Path):
    candidates = list(condition_dir.rglob("results*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    rows = []
    for condition_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not condition_dir.is_dir():
            continue

        results_path = find_results_json(condition_dir)
        if results_path is None:
            print(f"WARNING: no results.json found under {condition_dir}, skipping")
            continue

        with open(results_path) as f:
            data = json.load(f)

        all_results = data.get("results", {})
        ifeval_results = all_results.get("ifeval", {})
        mmlu_results = all_results.get("mmlu", {})

        if not ifeval_results and not mmlu_results:
            print(f"WARNING: no 'ifeval' or 'mmlu' key in {results_path}, skipping")
            continue

        rank, placement = parse_condition(condition_dir.name)
        row = {
            "condition": condition_dir.name,
            "rank": rank,
            "placement": placement,
        }
        for metric in IFEVAL_METRICS:
            short_name = "ifeval_" + metric.split(",")[0]
            row[short_name] = ifeval_results.get(metric, None)

        row["mmlu_acc"] = mmlu_results.get(MMLU_METRIC, None)

        rows.append(row)
        print(f"Parsed: {condition_dir.name}")

    if not rows:
        print("No results parsed. Check OUTPUT_ROOT path and that runs completed.")
        return

    fieldnames = (
        ["condition", "rank", "placement"]
        + ["ifeval_" + m.split(",")[0] for m in IFEVAL_METRICS]
        + ["mmlu_acc"]
    )
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written to {CSV_OUT}")


if __name__ == "__main__":
    main()
