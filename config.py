"""
Central configuration for the LoRA-Safety-Fine-Tuning pipeline.

Edit USERNAME below (and the *_ROOT paths, if your directory layout differs
from the default CSF3 scratch structure this project was built on) to match
your own cluster account before running anything in this repo. Every script
imports its paths from here instead of hardcoding them, so this is the only
file you should need to touch to get things running on your own setup.
"""

from pathlib import Path

# Your cluster username / account. Used to build the default scratch paths
# below — change this first.
USERNAME = "your_username_here"

# Root directories. Defaults follow the CSF3 scratch layout
# (/scratch/<username>/...) this pipeline was developed against. Override
# any of these directly if your own layout is different.

# Raw source datasets + preprocessing outputs (pku_train.jsonl,
# hh_train.jsonl, bt_train.jsonl, wg_train.parquet, test.jsonl, and the
# preprocessed.jsonl / preprocessed_clean.jsonl / annotated_train.xlsx /
# train.jsonl files produced along the way).
DATA_ROOT = Path(f"/scratch/{USERNAME}/data")

# Base (pretrained, non-fine-tuned) model checkpoints, one subfolder per
# model family, e.g. BASE_MODEL_ROOT / "falcon7b".
BASE_MODEL_ROOT = Path(f"/scratch/{USERNAME}/base_models")

# LoRA adapter checkpoints, one subfolder per condition, e.g.
# LORA_ROOT / "falcon7b_r4_early".
LORA_ROOT = Path(f"/scratch/{USERNAME}/lora_models")

# Full (non-LoRA) fine-tune checkpoints, one subfolder per model family,
# e.g. FULL_FT_ROOT / "falcon_full".
FULL_FT_ROOT = Path(f"/scratch/{USERNAME}/fullSFT_models")

# Capabilities evaluation (IFEval/MMLU) results, one subfolder per model
# family, e.g. EVAL_OUTPUT_ROOT / "falcon".
EVAL_OUTPUT_ROOT = Path(f"/scratch/{USERNAME}/eval_results")
