"""
Full-parameter fine-tuning, parametrised by model.

Replaces Mistral_Full_FineTune.py, Qwen_Full_FineTune.py, Falcon_Full_Finetune.py,
which were three identical copies of this script differing only in model_id /
output_dir.

Usage
-----
    python full_finetune.py --model mistral7b
    python full_finetune.py --model qwen7b --learning_rate 1e-5
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=config.MODEL_NAMES)
    p.add_argument("--data_file", default=None, help=f"default: {config.TRAIN_FILE}")
    p.add_argument("--output_dir", default=None, help="default: <FULL_FT_ROOT>/<model>_full")
    p.add_argument("--model_id", default=None, help="override the base model path")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=2e-5)
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
    output_dir = str(Path(args.output_dir) if args.output_dir else config.full_output_dir(args.model))

    print(f"Condition: model={args.model} full-parameter fine-tune", flush=True)
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

    # Gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print("Model loaded with gradient checkpointing enabled", flush=True)

    # ----------------------------
    # TRAINING CONFIG
    # ----------------------------
    config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        max_length=args.max_length,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    # ----------------------------
    # TRAINER
    # ----------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=config,
    )

    # ----------------------------
    # TRAIN
    # ----------------------------
    print("Starting full SFT training...", flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    trainer.train()

    # ----------------------------
    # SAVE MODEL
    # ----------------------------
    print("Saving model...", flush=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("DONE:", output_dir, flush=True)


if __name__ == "__main__":
    main()
