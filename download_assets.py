"""
One-stop download script for everything this pipeline needs before you can
run a single training/eval job: the three base model checkpoints, the
LlamaGuard safety checkpoint, the four raw training-data sources, and the
IFEval/MMLU eval-task data

Run this ONCE, with internet access, before submitting any GPU jobs.
Everything is written to the paths config.py already defines
(BASE_MODEL_ROOT, LLAMA_GUARD_MODEL_ROOT, DATA_ROOT), so set USERNAME (and
any *_ROOT overrides) in config.py first.

----
mistralai/* and meta-llama/* repos on the Hub are gated: you must accept
each repo's license on huggingface.co with the account you're downloading
as, THEN either run `huggingface-cli login` once, or export a token:

    export HF_TOKEN=hf_...

WildGuardMix is also gated and is NOT downloaded by this script; request
access at https://huggingface.co/datasets/allenai/wildguardmix with the
account you'll use, then once approved either add it back into
download_datasets() below or drop the file at
DATA_ROOT/wg_train.parquet yourself before running preprocess_train.py.

Repo IDs
--------
The exact checkpoint below is a project-specific choice, not something
this script can infer; double check the four MODEL_ID / *_REPO constants
below match what you actually fine-tuned / evaluated against before
relying on this for anything beyond a fresh setup. WildGuardMix is gated
and deliberately not handled here (see above).

Usage
-----
    python download_assets.py                # download everything
    python download_assets.py --models-only
    python download_assets.py --data-only
    python download_assets.py --eval-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Make config.py (at the repo root) importable regardless of where this
# script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ----------------------------------------------------------------------------
# Model checkpoints -- VERIFY these match what you actually used.
# ----------------------------------------------------------------------------
FALCON_REPO = "tiiuae/falcon-7b-instruct"
MISTRAL_REPO = "mistralai/Mistral-7B-Instruct-v0.3"
QWEN_REPO = "Qwen/Qwen2.5-7B-Instruct"
LLAMAGUARD_REPO = "meta-llama/Llama-Guard-4-12B"

MODEL_TARGETS = [
    (FALCON_REPO, config.BASE_MODEL_ROOT / "falcon7b"),
    (MISTRAL_REPO, config.BASE_MODEL_ROOT / "mistral7b"),
    (QWEN_REPO, config.BASE_MODEL_ROOT / "qwen7b"),
    (LLAMAGUARD_REPO, config.LLAMA_GUARD_MODEL_ROOT),
]

# Only pull safetensors + the small config/tokenizer files, not every
# duplicate weight format (.bin/.pth/etc) a repo might also ship -- saves
# a lot of bandwidth/disk on 7-12B checkpoints. If a repo you point this
# at doesn't ship safetensors, drop this filter for that repo.
MODEL_ALLOW_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "*.json", "*.model", "*.txt", "tokenizer*"]

# ----------------------------------------------------------------------------
# Raw training-data sources -- VERIFY split/config names match your setup.
# preprocess_train.py expects exactly these local filenames under DATA_ROOT,
# and these specific fields on each row (checked below after download).
# ----------------------------------------------------------------------------
PKU_REPO, PKU_SPLIT = "PKU-Alignment/PKU-SafeRLHF", "train"
PKU_REQUIRED_COLS = {"prompt", "prompt_source", "safer_response_id",
                      "is_response_0_safe", "is_response_1_safe", "response_0", "response_1"}

HH_REPO, HH_SPLIT = "Anthropic/hh-rlhf", "train"
HH_REQUIRED_COLS = {"chosen"}

BT_REPO, BT_SPLIT = "PKU-Alignment/BeaverTails", "330k_train"
BT_REQUIRED_COLS = {"prompt", "response", "is_safe"}

# Held-out BeaverTails split for eval use -- not consumed by any existing
# script in this repo yet, just downloaded alongside the training split.
BT_EVAL_SPLIT = "330k_test"

# WildGuardMix is gated on the Hub and is deliberately NOT downloaded here --
# see the Auth note in the module docstring above.


def _check_columns(ds, required, label):
    missing = required - set(ds.column_names)
    if missing:
        raise RuntimeError(
            f"{label}: downloaded dataset is missing expected column(s) {missing}. "
            f"This usually means the repo/split/config constant at the top of this "
            f"script doesn't match the dataset version preprocess_train.py was written "
            f"against -- check {label}_REPO / {label}_SPLIT above."
        )


def download_models(dry_run=False):
    from huggingface_hub import snapshot_download

    if not (os.environ.get("HF_TOKEN") or Path("~/.cache/huggingface/token").expanduser().exists()):
        print("WARNING: no HF_TOKEN set and no saved `huggingface-cli login` token found. "
              "mistralai/* and meta-llama/* are gated repos -- this will fail without auth.")

    for repo_id, local_dir in MODEL_TARGETS:
        if (local_dir / "config.json").exists():
            print(f"Skipping {repo_id}: already present at {local_dir}")
            continue
        print(f"Downloading {repo_id} -> {local_dir}")
        if dry_run:
            continue
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            allow_patterns=MODEL_ALLOW_PATTERNS,
        )
    print("Models done.\n")


def download_datasets(dry_run=False):
    from datasets import load_dataset

    config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    pku_file = config.DATA_ROOT / "pku_train.jsonl"
    hh_file = config.DATA_ROOT / "hh_train.jsonl"
    bt_file = config.DATA_ROOT / "bt_train.jsonl"
    bt_eval_file = config.DATA_ROOT / "bt_test.jsonl"

    jobs = [
        ("PKU", PKU_REPO, None, PKU_SPLIT, pku_file, PKU_REQUIRED_COLS, "json"),
        ("HH", HH_REPO, None, HH_SPLIT, hh_file, HH_REQUIRED_COLS, "json"),
        ("BT", BT_REPO, None, BT_SPLIT, bt_file, BT_REQUIRED_COLS, "json"),
        ("BT-eval", BT_REPO, None, BT_EVAL_SPLIT, bt_eval_file, BT_REQUIRED_COLS, "json"),
    ]

    print("Skipping WildGuardMix: it's gated on the Hub. Request access at "
          "https://huggingface.co/datasets/allenai/wildguardmix, then place the "
          f"approved file at {config.DATA_ROOT / 'wg_train.parquet'} yourself "
          "before running preprocess_train.py.\n")

    for label, repo_id, ds_config, split, out_path, required_cols, fmt in jobs:
        if out_path.exists():
            print(f"Skipping {label}: already present at {out_path}")
            continue
        print(f"Downloading {repo_id}" + (f" [{ds_config}]" if ds_config else "") + f" ({split}) -> {out_path}")
        if dry_run:
            continue
        ds = load_dataset(repo_id, ds_config, split=split) if ds_config else load_dataset(repo_id, split=split)
        _check_columns(ds, required_cols, label)
        if fmt == "parquet":
            ds.to_parquet(str(out_path))
        else:
            ds.to_json(str(out_path), orient="records", lines=True)
        print(f"  wrote {len(ds)} rows")
    print("Datasets done.\n")


def download_eval_tasks(dry_run=False):
    print("Resolving and downloading task data for: ifeval, mmlu ...")
    if dry_run:
        return
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager()
    task_dict = task_manager.load_task_or_group(["ifeval", "mmlu"])
    for task_name, task_obj in task_dict.items():
        print(f"  - {task_name}: dataset loaded ({len(task_obj.dataset) if hasattr(task_obj, 'dataset') else 'group'})")
    print("Eval task data done (cached under ~/.cache/huggingface/).\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-only", action="store_true")
    p.add_argument("--data-only", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print what would be downloaded, download nothing")
    args = p.parse_args()

    run_models = args.models_only or not (args.data_only or args.eval_only)
    run_data = args.data_only or not (args.models_only or args.eval_only)
    run_eval = args.eval_only or not (args.models_only or args.data_only)

    if run_models:
        download_models(dry_run=args.dry_run)
    if run_data:
        download_datasets(dry_run=args.dry_run)
    if run_eval:
        download_eval_tasks(dry_run=args.dry_run)

    print("All requested downloads complete." if not args.dry_run else "Dry run complete -- nothing was downloaded.")


if __name__ == "__main__":
    main()
