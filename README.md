# LoRA-Safety-Fine-Tuning

## Overview

Master's dissertation project evaluating how LoRA rank and adapter placement affect safety alignment outcomes and general capabilities (IFEval, MMLU) across Falcon-7B, Mistral-7B, and Qwen-7B.

Models are evaluated on their ability to produce safe responses and accurate harm classifications in response to the evaluation prompts, whilst retaining their general capabilities. Additional attribution analysis is conducted on three selected Mistral-7B models by estimating word-level Shapley values across a sample of evaluation prompts to assess which words most contributed towards the models' harm classifications. 

## Setup

Clone the repo and create a virtual environment:

```
git clone https://github.com/ellarnoble/LoRA-Safety-Fine-Tuning.git
cd LoRA-Safety-Fine-Tuning
python -m venv venv
```

Activate the venv and install dependencies using the `requirements.txt` file . On a GPU cluster, install `torch` first with the wheel matching your CUDA version (see
[pytorch.org/get-started](https://pytorch.org/get-started/locally/)) before installing the rest, since the plain `pip install torch` in requirements.txt will otherwise pull a CPU-only build:

```
pip install torch --index-url <your CUDA-specific index URL>
pip install -r requirements.txt
```

Then open `config.py` at the repo root and set `USERNAME` (and any of the `*_ROOT` paths, if your directory layout differs from the CSF3 scratch structure this project was built on) to match your own account. Every script in this repo imports its paths from `config.py`, it's the only file you should need to edit to get things running on your own machine or cluster account.

## Pipeline

1. **Download assets.** `python download_assets.py` pulls the three base checkpoints, the LlamaGuard checkpoint, and the raw PKU-SafeRLHF/HH-RLHF/BeaverTails data into the paths set in `config.py`. WildGuardMix is gated and not fetched automatically; request access at huggingface.co/datasets/allenai/wildguardmix and place the approved file at `DATA_ROOT/wg_train.parquet`.

2. **Preprocess and annotate training data.** `data/train/preprocessing/preprocess_train.py` merges, deduplicates, and cleans the four raw sources into `preprocessed.jsonl`. `GPT_annotation.py` (requires `OPENAI_API_KEY`) then adds GPT-based topic/harm annotations to produce the final `train.jsonl` used below.

3. **Fine-tune.** `finetuning_scripts/submit_all_lora.sh` trains all 36 LoRA conditions (3 models × 4 ranks × 3 placements); `submit_all_fullFT.sh` trains the 3 full fine-tunes. Both skip conditions that are already trained, so a resubmitted job resumes where it left off. `LoRA_finetune.py`/`Full_finetune.py` can also be called directly for a single condition.

4. **Generate responses.** `submit_all_generate.sh` runs every trained condition plus the untuned baselines against the held-out test set, writing one `<model>_<condition>_responses.jsonl` per condition.

5. **Safety evaluation.** `submit_all_evaluate.sh` scores each response file with LlamaGuard and compares its verdict against the model's own harm-classification token (HCT), producing the SRR and HCT-accuracy metrics.

6. **Capabilities evaluation.** `capabilities_evaluation/scripts/run_all_evals.py` runs IFEval and MMLU via lm-evaluation-harness across all conditions; the `aggregate_*_eval.py` scripts collect the results into `mmlu_ifeval_results.xlsx`.

7. **Attribution analysis.** `attribution_analysis/scripts/owen_shapley_attribution.py <condition>` computes word-level Shapley attributions for the harm-classification decision on the three selected Mistral-7B conditions; `plot_attributions.py` renders the comparison figures.

8. **Statistical analysis.** The R scripts in `safety_analysis/scripts/` take the aggregated safety and capabilities results and produce the omnibus tests and plots reported in the dissertation.
