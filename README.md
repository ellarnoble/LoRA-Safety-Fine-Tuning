# LoRA-Safety-Fine-Tuning

Master's dissertation project evaluating how LoRA rank and adapter placement affect
safety alignment outcomes and general capabilities (IFEval, MMLU) across
Falcon-7B, Mistral-7B, and Qwen-7B.

## Repo layout

```
config.py                       
requirements.txt
data/                            
  test/
    test.jsonl
  train/
    annotated_train.xlsx
    train.jsonl
preprocessing/
  preprocess_train.py
  GPT_annotation.py               
capabilities_evaluation/
  scripts/
    predownload_evaltasks.py      
    falcon_eval.py
    mistral_eval.py
    qwen_eval.py
    aggregate_falcon_eval.py
    aggregate_mistral_eval.py
    aggregate_qwen_eval.py
    run_all_evals.py             
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
script in this repo imports its paths from `config.py`, it's the only file
you should need to edit to get things running on your own machine or
cluster account.

## Pipeline
