"""
Run IFEval + MMLU across the base Mistral-7B model and all 9 LoRA
conditions (3 ranks x 3 placements).

Calls lm-evaluation-harness as a subprocess for each condition, with
both tasks requested in a single lm_eval call (this is more efficient
than separate calls since the model only needs to be loaded once per
condition).

Skips any condition whose results already exist, so it's safe to
resubmit if the SLURM job times out partway through.
"""

import subprocess
import sys
from pathlib import Path

# Make config.py (at the repo root) importable regardless of where this
# script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BASE_MODEL_ROOT, LORA_ROOT, FULL_FT_ROOT, EVAL_OUTPUT_ROOT

# ---- Paths (edit config.py at the repo root, not here) ----
BASE_MODEL = str(BASE_MODEL_ROOT / "mistral7b")
FULL_FT_MODEL = FULL_FT_ROOT / "mistral_full"
OUTPUT_ROOT = EVAL_OUTPUT_ROOT / "mistral"

RANKS = [1, 4, 16, 64]
PLACEMENTS = ["early", "middle", "late"]

# IFEval = instruction-following compliance (rule-based scoring)
# MMLU   = general knowledge / reasoning (multiple-choice, loglikelihood scoring)
# Running both in one lm_eval call per condition so the model is loaded once.
TASKS = "ifeval,mmlu"
BATCH_SIZE = "auto"
SEED = "42"


def run_eval(model_args: str, out_path: Path, tag: str):
    print("==============================================")
    print(f"Running IFEval for: {tag}")
    print(f"model_args: {model_args}")
    print("==============================================")

    if (out_path / "results.json").exists() or list(out_path.rglob("results*.json")):
        print(f"Results already exist at {out_path}, skipping. Delete to force re-run.")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", TASKS,
        "--batch_size", BATCH_SIZE,
        "--output_path", str(out_path),
        "--log_samples",
        "--seed", SEED,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: lm_eval exited with code {result.returncode} for {tag}")
    else:
        print(f"Done: {tag}")
    print()


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Base model (no adapter) — reference point
    run_eval(
        model_args=f"pretrained={BASE_MODEL},dtype=bfloat16",
        out_path=OUTPUT_ROOT / "base",
        tag="mistral7b_base (ifeval+mmlu)",
    )

    # 2. Full fine-tune condition — standalone merged model, no peft= needed.
    # Skipped gracefully if not yet trained, so this script stays runnable
    # before the full fine-tune checkpoint exists.
    if FULL_FT_MODEL.is_dir() and (FULL_FT_MODEL / "config.json").exists():
        run_eval(
            model_args=f"pretrained={FULL_FT_MODEL},dtype=bfloat16",
            out_path=OUTPUT_ROOT / "full_ft",
            tag="mistral7b_full_ft (ifeval+mmlu)",
        )
    else:
        print(f"WARNING: full fine-tune checkpoint not found at {FULL_FT_MODEL}, skipping full_ft condition")
        print()

    # 3. All LoRA conditions (rank x placement)
    for rank in RANKS:
        for placement in PLACEMENTS:
            condition = f"r{rank}_{placement}"
            adapter_dir = LORA_ROOT / f"mistral7b_{condition}"

            if not adapter_dir.is_dir():
                print(f"WARNING: {adapter_dir} not found, skipping {condition}")
                continue
            if not (adapter_dir / "adapter_model.safetensors").exists():
                print(f"WARNING: no adapter_model.safetensors in {adapter_dir}, skipping {condition}")
                continue

            model_args = f"pretrained={BASE_MODEL},peft={adapter_dir},dtype=bfloat16"
            out_path = OUTPUT_ROOT / f"mistral7b_{condition}"

            run_eval(model_args=model_args, out_path=out_path, tag=f"mistral7b_{condition}")

    print(f"All IFEval + MMLU runs complete. Results in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
