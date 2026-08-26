"""
LoRA fine-tuning, parametrised by model / rank / adapter placement.

Usage
-----
    python lora_finetune.py --model mistral7b --rank 4 --placement early
    python lora_finetune.py --model qwen7b --rank 16 --placement late --alpha 32
    python lora_finetune.py --model falcon7b --rank 64 --placement middle \\
        --data_file /scratch/me/data/cleaned/train.jsonl

Layer ranges for each (model, placement) pair are looked up automatically
from config.MODEL_REGISTRY
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=config.MODEL_NAMES)
    p.add_argument("--rank", required=True, type=int, choices=config.RANKS)
    p.add_argument("--placement", required=True, choices=config.PLACEMENTS)
    p.add_argument("--alpha", type=int, default=config.DEFAULT_LORA_ALPHA,
                    help=f"lora_alpha (default: fixed at {config.DEFAULT_LORA_ALPHA} for every rank, "
                         "matching the original pilot run)")
    p.add_argument("--lora_dropout", type=float, default=config.DEFAULT_LORA_DROPOUT)
    p.add_argument("--target_modules", nargs="+", default=None,
                    help="default: model-specific target_modules from config.MODEL_REGISTRY")
    p.add_argument("--data_file", default=None, help=f"default: {config.TRAIN_FILE}")
    p.add_argument("--output_dir", default=None,
                    help="default: <LORA_ROOT>/<model>_r<rank>_<placement>")
    p.add_argument("--model_id", default=None, help="override the base model path")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=float, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    spec = config.MODEL_REGISTRY[args.model]

    model_id = str(Path(args.model_id) if args.model_id else spec.base_path)
    data_file = str(Path(args.data_file) if args.data_file else config.TRAIN_FILE)
    output_dir = str(Path(args.output_dir) if args.output_dir
                      else config.lora_output_dir(args.model, args.rank, args.placement))
    target_modules = args.target_modules or list(spec.target_modules)

    start, end, layers_to_transform = config.placement_layers(args.model, args.placement)
    print(f"Condition: model={args.model} rank={args.rank} placement={args.placement} "
          f"-> layers {start}-{end} ({len(layers_to_transform)} layers)", flush=True)
    print(f"model_id={model_id}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # ----------------------------
    # LOAD TOKENIZER
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ----------------------------
    # LOAD AND FORMAT DATASET
    # ----------------------------
    print("Loading training data...", flush=True)
    dataset = load_dataset("json", data_files=data_file, split="train")

    def format_example(example):
        hct = str(example["hct"])
        return {
            "prompt": config.format_prompt(example["prompt"], tokenizer),
            "completion": f"{hct}\n{example['response']}",
        }

    dataset = dataset.map(format_example)
    print(f"Formatted {len(dataset)} training examples", flush=True)

    # ----------------------------
    # LOAD MODEL
    # ----------------------------
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # ----------------------------
    # LORA CONFIG
    # ----------------------------
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ----------------------------
    # TRAINING CONFIG
    # ----------------------------
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    # ----------------------------
    # TRAIN
    # ----------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=sft_config,
    )

    print("Starting LoRA fine-tuning...", flush=True)
    trainer.train()

    # ----------------------------
    # SAVE
    # ----------------------------
    print("Saving adapter weights...", flush=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"DONE: Model saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
