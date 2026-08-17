"""
Owen-Shapley word-importance attribution for the harm-classification-token
(hct) prediction of a fine-tuned model.

For each prompt in a condition's "<model>_<label>_100_selected.jsonl" file
(under config.ATTRIBUTION_PROMPTS_ROOT), computes a per-word Shapley value
showing how much each word in the prompt contributes to the model's hct=1
(unsafe) probability. Uses pyDVL's Owen sampling estimator over an
HCTTokenUtility that re-runs the model on word subsets.

Usage
-----
    python owen_shapley_attribution.py r64_middle
    python owen_shapley_attribution.py full
    python owen_shapley_attribution.py r1_late

Conditions below are for mistral7b only, matching the three checkpoints this
analysis was run against. Add entries to CONDITIONS to cover more.
"""

import sys
from pathlib import Path

# Make config.py (at the repo root) importable regardless of where this
# script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    attribution_output_dir,
    attribution_prompts_file,
    format_prompt,
    full_output_dir,
    lora_output_dir,
)

import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline

from pydvl.valuation import AntitheticOwenSampler, RankCorrelation, ShapleyValuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.samplers.owen import GridOwenStrategy
from pydvl.valuation.utility.base import UtilityBase

torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)

MODEL = "mistral7b"

CONDITIONS = {
    "r64_middle": dict(
        model_dir=lora_output_dir(MODEL, 64, "middle"),
        output_dir=attribution_output_dir(MODEL, "lora", 64, "middle"),
        prompts_file=attribution_prompts_file(MODEL, "lora", 64, "middle"),
    ),
    "full": dict(
        model_dir=full_output_dir(MODEL),
        output_dir=attribution_output_dir(MODEL, "full"),
        prompts_file=attribution_prompts_file(MODEL, "full"),
    ),
    "r1_late": dict(
        model_dir=lora_output_dir(MODEL, 1, "late"),
        output_dir=attribution_output_dir(MODEL, "lora", 1, "late"),
        prompts_file=attribution_prompts_file(MODEL, "lora", 1, "late"),
    ),
}

if len(sys.argv) != 2 or sys.argv[1] not in CONDITIONS:
    sys.exit(f"Usage: python {Path(__file__).name} <{'|'.join(CONDITIONS)}>")

CONDITION = sys.argv[1]
cfg = CONDITIONS[CONDITION]
model_dir = str(cfg["model_dir"])
output_dir = cfg["output_dir"]
output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
generator = pipeline(
    "text-generation",
    model=model_dir,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device=0,
)
if generator.tokenizer.pad_token_id is None:
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

TOKEN_ID_0 = tokenizer.encode("0", add_special_tokens=False)[-1]
TOKEN_ID_1 = tokenizer.encode("1", add_special_tokens=False)[-1]


def hct_probability(prompt_text):
    formatted = format_prompt(prompt_text, tokenizer)
    inputs = tokenizer(formatted, return_tensors="pt").to(generator.model.device)
    with torch.no_grad():
        logits = generator.model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits[[TOKEN_ID_0, TOKEN_ID_1]], dim=-1)
    return probs[1].item()


class HCTTokenUtility(UtilityBase):
    def __init__(self, words, hct_prob_fn):
        self.words = words
        self.hct_prob_fn = hct_prob_fn
        self._training_data = None

    def __call__(self, sample):
        if sample is None or sample.subset is None or len(sample.subset) == 0:
            present_words = []
        else:
            present_words = [self.words[i] for i in sorted(sample.subset)]
        return self.hct_prob_fn(" ".join(present_words))


def compute_shapley_importance(
    prompt_text,
    hct_prob_fn=hct_probability,
    n_samples_outer=64,
    n_samples_inner=2,
    rtol=1e-3,
    burn_in=32,
):
    words = prompt_text.split()
    n = len(words)
    x_dummy = np.arange(n).reshape(-1, 1)
    y_dummy = np.zeros(n)
    unique_names = [f"{w}_{i}" for i, w in enumerate(words)]
    dataset = Dataset(x_dummy, y_dummy, data_names=unique_names)
    utility = HCTTokenUtility(words, hct_prob_fn)
    sampler = AntitheticOwenSampler(
        outer_sampling_strategy=GridOwenStrategy(n_samples_outer=n_samples_outer),
        n_samples_inner=n_samples_inner,
    )
    stopping = RankCorrelation(rtol=rtol, burn_in=burn_in)
    valuation = ShapleyValuation(utility, sampler, stopping)
    valuation.fit(dataset)
    result = valuation.result
    return pd.DataFrame(
        {
            "word": [words[i] for i in result.indices],
            "position": result.indices,
            "shapley_value": result.values,
            "variance": result.variances,
            "n_updates": result.counts,
        }
    ).sort_values("shapley_value", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
with open(cfg["prompts_file"], encoding="utf-8") as f:
    EXAMPLES = [json.loads(line) for line in f]
print(f"Loaded {len(EXAMPLES)} prompts from {cfg['prompts_file']}", flush=True)

for i, ex in enumerate(EXAMPLES):
    print(f"...Index {ex['index']} (hct={ex['hct']})...", flush=True)
    df = compute_shapley_importance(ex["prompt"])
    print(df.head(15).to_string(index=False), flush=True)
    df.to_csv(output_dir / f"index{ex['index']}_shapley.csv", index=False)

print(f"DONE. {len(EXAMPLES)} prompts processed. Results saved under {output_dir}", flush=True)
