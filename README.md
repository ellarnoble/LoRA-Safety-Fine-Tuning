# LoRA-Safety-Fine-Tuning

Dissertation project evaluating how LoRA rank and adapter placement affect
safety-tuning outcomes and general capabilities (IFEval, MMLU) across
Falcon-7B, Mistral-7B, and Qwen-7B.

## Repo layout

```
config.py                          # edit this before running anything
requirements.txt
data/                               # committed, ready-to-use data artifacts
  test/test.jsonl
  train/annotated_train.xlsx
  train/train.jsonl
preprocessing/
  preprocess_train.py               # source datasets -> preprocessed.jsonl
capabilities_evaluation/
  scripts/
    predownload_evaltasks.py        # run once, CSF3 login node only
    falcon_eval.py
    mistral_eval.py
    qwen_eval.py
    aggregate_falcon_eval.py
    aggregate_mistral_eval.py
    aggregate_qwen_eval.py
    run_all_evals.py                # orchestrates all six scripts above
```

## Setup

Clone the repo and create a virtual environment:

```
git clone https://github.com/ellarnoble/LoRA-Safety-Fine-Tuning.git
cd LoRA-Safety-Fine-Tuning
python -m venv venv
```

Activate it (`.\venv\Scripts\Activate.ps1` on Windows PowerShell, `source venv/bin/activate`
on macOS/Linux), then install dependencies. On a GPU cluster, install `torch`
first with the wheel matching your CUDA version (see
[pytorch.org/get-started](https://pytorch.org/get-started/locally/)) before
installing the rest, since the plain `pip install torch` in requirements.txt
will otherwise pull a CPU-only build:

```
pip install torch --index-url <your CUDA-specific index URL>
pip install -r requirements.txt
```

Then open `config.py` at the repo root and set `USERNAME` (and any of the
`*_ROOT` paths, if your directory layout differs from the CSF3 scratch
structure this project was built on) to match your own account. Every
script in this repo imports its paths from `config.py` — it's the only file
you should need to edit to get things running on your own machine or
cluster account.

## Pipeline

1. **Preprocessing** — `python preprocessing/preprocess_train.py` loads the
   raw PKU-SafeRLHF, Anthropic HH-RLHF, BeaverTails, and WildGuard datasets
   from `DATA_ROOT` (set in `config.py`), filters, standardises, dedupes,
   removes any prompts overlapping with `DATA_ROOT/test.jsonl`, removes
   near-duplicate/paraphrased prompts, and writes the result to
   `DATA_ROOT/preprocessed.jsonl`.

2. **Safety annotation** (not included in this repo yet) — a GPT-based
   annotation script run on `preprocessed.jsonl` that labels each example's
   `response_safety` and produces `annotated_train.xlsx`.

3. **LoRA / full fine-tuning** — train your adapters/checkpoints against
   `annotated_train.xlsx` (or its derived training file). Training scripts
   aren't part of this repo checkout.

4. **Capabilities evaluation** — once you have base models, LoRA adapters,
   and/or full fine-tune checkpoints in place under the paths configured in
   `config.py`, run:

   ```
   python capabilities_evaluation/scripts/run_all_evals.py
   ```

   from the CSF3 login node with `--predownload` the first time (compute
   nodes can't reach HuggingFace Hub), or omit it if IFEval/MMLU are already
   cached. This runs IFEval + MMLU across the base model, full fine-tune (if
   present), and every LoRA rank/placement condition, for all three model
   families, then aggregates each into a CSV and merges them into one
   combined summary. Pass `--models falcon qwen` to run a subset, or see
   `python run_all_evals.py --help` for the rest of the options. Each
   individual script (`falcon_eval.py`, `aggregate_falcon_eval.py`, etc.) can
   also be run standalone if you'd rather not use the orchestrator.

## Data already in this repo

The `data/` folder contains the outputs of the pipeline as already run for
this dissertation: the held-out test split, the GPT-annotated training data
(`annotated_train.xlsx`), and the final paraphrase-deduplicated training set
(`train.jsonl`). You don't need to regenerate these to reproduce the
evaluation results — only the preprocessing and capabilities evaluation
steps above.
