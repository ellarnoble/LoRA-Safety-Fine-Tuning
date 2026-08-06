"""
Orchestrates the full Capabilities Evaluation pipeline for one or more
model families (Falcon-7B, Mistral-7B, Qwen-7B) with a single command,
instead of running six-plus scripts by hand in the right order.

This is a thin wrapper around the existing scripts in this folder
(falcon_eval.py, mistral_eval.py, qwen_eval.py and their aggregate_*
counterparts) — it does not change what any of them do. It just:

  - runs each model's eval step, then its aggregate step, in order
  - times every stage and prints clear section headers
  - if one model's eval step fails, keeps going with the remaining
    models instead of aborting the whole run (each per-model script
    already skips conditions whose results exist, so re-running after
    a partial failure is cheap)
  - merges the three per-model summary CSVs into one combined CSV with
    a "model" column, so you don't have to open three files
  - prints a final OK/FAILED table per model/stage and exits non-zero
    if anything failed, so it plays nicely in a SLURM job script

Usage:
    # Run everything: all three models, eval + aggregate, then merge
    python run_all_evals.py

    # Only specific model families
    python run_all_evals.py --models falcon qwen

    # Re-run only the aggregation + merge step (e.g. after manually
    # re-running one failed condition) without repeating all the evals
    python run_all_evals.py --aggregate-only

    # Only run the eval step, skip aggregation/merge
    python run_all_evals.py --eval-only

    # Also run predownload_evaltasks.py first. Only useful on the CSF3
    # LOGIN node before submitting the GPU job — see that script's
    # docstring; compute nodes can't reach HuggingFace Hub.
    python run_all_evals.py --predownload
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

MODELS = {
    "falcon": {
        "eval": SCRIPT_DIR / "falcon_eval.py",
        "aggregate": SCRIPT_DIR / "aggregate_falcon_eval.py",
        "output_root": Path("/scratch/f42827en/eval_results/falcon"),
    },
    "mistral": {
        "eval": SCRIPT_DIR / "mistral_eval.py",
        "aggregate": SCRIPT_DIR / "aggregate_mistral_eval.py",
        "output_root": Path("/scratch/f42827en/eval_results/mistral"),
    },
    "qwen": {
        "eval": SCRIPT_DIR / "qwen_eval.py",
        "aggregate": SCRIPT_DIR / "aggregate_qwen_eval.py",
        "output_root": Path("/scratch/f42827en/eval_results/qwen"),
    },
}

COMBINED_CSV = SCRIPT_DIR / "ifeval_mmlu_summary_all_models.csv"


def run_step(script_path: Path, label: str) -> bool:
    """Run one script as a subprocess, streaming its output live. Returns True on success."""
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    start = time.time()
    result = subprocess.run([sys.executable, str(script_path)])
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"--- {label}: {status} in {elapsed / 60:.1f} min ---")
    return ok


def merge_csvs(models_run):
    """Combine each model's ifeval_mmlu_summary.csv into one file with a 'model' column."""
    all_rows = []
    fieldnames = None
    for model in models_run:
        csv_path = MODELS[model]["output_root"] / "ifeval_mmlu_summary.csv"
        if not csv_path.exists():
            print(f"  (no summary CSV for {model} at {csv_path}, skipping it in the merge)")
            continue
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = ["model"] + reader.fieldnames
            for row in reader:
                all_rows.append({"model": model, **row})

    if not all_rows:
        print("  No per-model summary CSVs found to merge.")
        return

    with open(COMBINED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Combined summary written to {COMBINED_CSV}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the IFEval+MMLU eval and aggregation pipeline across model families."
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS.keys(), default=list(MODELS.keys()),
        help="Which model families to run (default: all three)",
    )
    parser.add_argument(
        "--predownload", action="store_true",
        help="Run predownload_evaltasks.py first (CSF3 login node only)",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip the eval step; only (re-)run aggregation and the CSV merge",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run eval only; skip aggregation and the CSV merge",
    )
    args = parser.parse_args()

    if args.aggregate_only and args.eval_only:
        parser.error("--aggregate-only and --eval-only can't be used together")

    overall_start = time.time()
    print(f"Capabilities evaluation pipeline started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Models: {', '.join(args.models)}")

    if args.predownload:
        ok = run_step(SCRIPT_DIR / "predownload_evaltasks.py", "Predownload IFEval/MMLU datasets")
        if not ok:
            print("Predownload reported a failure — continuing anyway, but eval runs may fail "
                  "if the datasets aren't already cached.")

    results = {}  # model -> {"eval": bool|None, "aggregate": bool|None}

    for model in args.models:
        results[model] = {"eval": None, "aggregate": None}

        if not args.aggregate_only:
            ok = run_step(MODELS[model]["eval"], f"{model.upper()} evaluation (IFEval + MMLU)")
            results[model]["eval"] = ok
            if not ok:
                print(f"  {model} eval step reported a failure — moving on to aggregate whatever "
                      f"conditions did complete, then continuing to the next model.")

        if not args.eval_only:
            ok = run_step(MODELS[model]["aggregate"], f"{model.upper()} results aggregation")
            results[model]["aggregate"] = ok

    if not args.eval_only:
        print(f"\n{'=' * 60}\nMerging per-model summaries\n{'=' * 60}")
        merge_csvs(args.models)

    elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}\nPipeline finished in {elapsed / 60:.1f} min\n{'=' * 60}")
    for model, stages in results.items():
        parts = [f"{stage}={'OK' if ok else 'FAILED'}" for stage, ok in stages.items() if ok is not None]
        print(f"  {model}: {', '.join(parts) if parts else '(skipped)'}")

    any_failed = any(ok is False for stages in results.values() for ok in stages.values())
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
