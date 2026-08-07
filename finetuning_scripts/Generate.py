"""
Generate model responses on the held-out test set, parametrised by model and
condition (baseline / full fine-tune / a specific LoRA rank+placement).

Replaces Mistral_Generate.py, Qwen_Generate.py, Falcon_Generate.py, which had
to be hand-edited (model_dir, output_file) for every one of the 14 conditions
per model.

Mistral and Qwen are generated through transformers' `pipeline("text-generation")`.
Falcon is generated through a direct `model.generate()` call instead, because its
<|user|>/<|assistant|>/<|system|> turn markers are plain text tokens (not special
tokens), so they survive `skip_special_tokens=True` and have to be truncated out
manually (see config.truncate_at_turn_marker). This script picks the right code
path automatically from config.MODEL_REGISTRY[model].generation_mode -- you don't
need to remember which model needs which treatment.

Usage
-----
    python generate.py --model mistral7b --condition baseline
    python generate.py --model qwen7b --condition full
    python generate.py --model falcon7b --condition lora --rank 16 --placement middle
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=config.MODEL_NAMES)
    p.add_argument("--condition", required=True, choices=["baseline", "full", "lora"])
    p.add_argument("--rank", type=int, choices=config.RANKS, default=None,
                    help="required when --condition lora")
    p.add_argument("--placement", choices=config.PLACEMENTS, default=None,
                    help="required when --condition lora")
    p.add_argument("--model_dir", default=None,
                    help="override the directory generate.py loads (default: derived from "
                         "--model/--condition/--rank/--placement via config.resolve_model_dir)")
    p.add_argument("--data_file", default=None, help=f"default: {config.TEST_FILE}")
    p.add_argument("--output_file", default=None,
                    help="default: <GENERATE_ROOT>/<model>/<model>_<condition_label>_responses.jsonl")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_paths(args):
    if args.condition == "lora" and (args.rank is None or args.placement is None):
        raise SystemExit("--rank and --placement are required when --condition lora")

    model_dir = str(Path(args.model_dir) if args.model_dir
                     else config.resolve_model_dir(args.model, args.condition, args.rank, args.placement))
    data_file = str(Path(args.data_file) if args.data_file else config.TEST_FILE)

    label = config.condition_label(args.condition, args.rank, args.placement)
    output_file = Path(args.output_file) if args.output_file \
        else config.generate_output_file(args.model, args.condition, args.rank, args.placement)
    return model_dir, data_file, output_file, label


# ----------------------------------------------------------------------------
# Mistral / Qwen: pipeline-based generation
# ----------------------------------------------------------------------------
def run_pipeline_generation(model_dir, df, prompts, output_file, batch_size, max_new_tokens):
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")

    generator = pipeline(
        "text-generation",
        model=model_dir,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device=0,
    )

    print("Model class:", generator.model.__class__.__name__, flush=True)
    try:
        active_adapters = generator.model.active_adapters()
    except (ValueError, AttributeError):
        active_adapters = "No adapters found"
    print("Active adapters:", active_adapters, flush=True)

    if generator.tokenizer.pad_token_id is None:
        generator.tokenizer.pad_token_id = generator.model.config.eos_token_id

    print("Model ready on:", generator.model.device, flush=True)

    print("Formatting prompts with chat template...", flush=True)
    formatted_prompts = [config.format_prompt(p, generator.tokenizer) for p in prompts]

    print("Starting generation...", flush=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(formatted_prompts), batch_size), mininterval=1):
            batch = formatted_prompts[start:start + batch_size]
            try:
                outputs = generator(batch, max_new_tokens=max_new_tokens, do_sample=False)
            except Exception as e:
                print(f"[BATCH ERROR] {start}: {e}", flush=True)
                for j in range(len(batch)):
                    idx = start + j
                    _write_error_row(f, df, idx)
                f.flush()
                continue

            for j, out in enumerate(outputs):
                idx = start + j
                try:
                    prompt = prompts[idx]
                    if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
                        text = out[0]["generated_text"]
                        text = config.strip_prompt(formatted_prompts[idx], text)
                    elif isinstance(out, dict) and "generated_text" in out:
                        text = out["generated_text"]
                        text = config.strip_prompt(formatted_prompts[idx], text)
                    else:
                        raise ValueError(f"Unexpected output type: {type(out)}")

                    hct, response_text = config.parse_hct(text)
                    _write_row(f, df, idx, prompt, hct, response_text)
                except Exception as e:
                    print(f"[WRITE ERROR] idx={idx}: {e}", flush=True)
                    _write_error_row(f, df, idx)
            f.flush()


# ----------------------------------------------------------------------------
# Falcon: manual generate() + turn-marker truncation
# ----------------------------------------------------------------------------
def run_manual_generation(model_dir, df, prompts, output_file, batch_size, max_new_tokens):
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    print("Model class:", model.__class__.__name__, flush=True)
    try:
        print("Active adapters:", model.active_adapters(), flush=True)
    except ValueError:
        print("Active adapters: None (baseline: no adapters loaded)", flush=True)
    print("Model ready on:", next(model.parameters()).device, flush=True)

    print("Formatting prompts with chat template...", flush=True)
    formatted_prompts = [config.format_prompt(p, tokenizer) for p in prompts]

    print("Starting generation...", flush=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(formatted_prompts), batch_size), mininterval=1):
            batch_prompts = formatted_prompts[start:start + batch_size]
            batch_size_actual = len(batch_prompts)

            try:
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to("cuda:0")

                input_lengths = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        eos_token_id=None,  # Prevent early stopping after HCT token
                        pad_token_id=tokenizer.pad_token_id,
                    )

                new_tokens = outputs[:, input_lengths:]
                decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            except Exception as e:
                print(f"[BATCH ERROR] {start}: {e}", flush=True)
                for j in range(batch_size_actual):
                    idx = start + j
                    _write_error_row(f, df, idx)
                f.flush()
                continue

            for j, text in enumerate(decoded):
                idx = start + j
                try:
                    prompt = prompts[idx]
                    text = config.truncate_at_turn_marker(text.strip())
                    hct, response_text = config.parse_hct(text)
                    _write_row(f, df, idx, prompt, hct, response_text)
                except Exception as e:
                    print(f"[WRITE ERROR] idx={idx}: {e}", flush=True)
                    _write_error_row(f, df, idx)
            f.flush()


def _write_row(f, df, idx, prompt, hct, response_text):
    result = {
        "index": config.safe_json(idx),
        "prompt": config.safe_json(prompt),
        "hct": hct,
        "response": config.safe_json(response_text),
        "source": config.safe_json(df.iloc[idx]["source"]),
    }
    f.write(json.dumps(result) + "\n")


def _write_error_row(f, df, idx):
    result = {
        "index": config.safe_json(idx),
        "prompt": config.safe_json(df.iloc[idx]["prompt"]),
        "hct": None,
        "response": None,
        "source": config.safe_json(df.iloc[idx]["source"]),
    }
    f.write(json.dumps(result) + "\n")


def main():
    args = parse_args()
    model_dir, data_file, output_file, label = resolve_paths(args)
    spec = config.MODEL_REGISTRY[args.model]

    print(f"Condition: model={args.model} condition={args.condition} label={label}", flush=True)
    print(f"model_dir={model_dir}", flush=True)
    print(f"output_file={output_file}", flush=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(output_file)

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    print("Checking CUDA...", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    print("Loading dataset...", flush=True)
    df = pd.read_json(data_file, lines=True)
    prompts = df["prompt"].tolist()
    print(f"Loaded {len(prompts)} prompts", flush=True)

    if spec.generation_mode == "pipeline":
        run_pipeline_generation(model_dir, df, prompts, output_file, args.batch_size, args.max_new_tokens)
    elif spec.generation_mode == "manual":
        run_manual_generation(model_dir, df, prompts, output_file, args.batch_size, args.max_new_tokens)
    else:
        raise ValueError(f"Unknown generation_mode {spec.generation_mode!r} for {args.model}")

    print("DONE:", output_file, flush=True)


if __name__ == "__main__":
    main()
