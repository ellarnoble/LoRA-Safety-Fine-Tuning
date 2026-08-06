from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd

USERNAME = "f42827en"
PKU_file = f"/scratch/{USERNAME}/data/pku_train.jsonl"
HH_file = f"/scratch/{USERNAME}/data/hh_train.jsonl"
BT_file = f"/scratch/{USERNAME}/data/bt_train.jsonl"
WG_file = f"/scratch/{USERNAME}/data/wg_train.parquet"

# Used for the leakage check in the final step below. Adjust if your
# test split lives somewhere else.
TEST_file = f"/scratch/{USERNAME}/data/test.jsonl"

MIN_RESPONSE_LENGTH = 25
# ----------------------------
# LOAD PKU DATASET
# ----------------------------
print("Loading PKU-SafeRLHF dataset...", flush=True)
PKU_dataset = load_dataset("json", data_files=PKU_file, split="train")

# Remove BeaverTails overlap
PKU_dataset = PKU_dataset.filter(lambda x: x["prompt_source"] != "beavertails")

# Keep only examples where the safer response is also actually safe
def keep_safe(example):
    sid = example["safer_response_id"]
    if sid == 0:
        return example["is_response_0_safe"]
    elif sid == 1:
        return example["is_response_1_safe"]
    return False

PKU_dataset = PKU_dataset.filter(keep_safe)

# Standardise schema
def format_pku(example):
    sid = example["safer_response_id"]
    response = example["response_0"] if sid == 0 else example["response_1"]
    return {
        "prompt": example["prompt"],
        "response": response,
        "source": "pku",
        "original_id": example.get("id", None)
    }

PKU_dataset = PKU_dataset.map(format_pku, remove_columns=PKU_dataset.column_names)

# Remove very short responses
PKU_dataset = PKU_dataset.filter(lambda x: len(x["response"].strip()) > MIN_RESPONSE_LENGTH)

print(f"PKU dataset size: {len(PKU_dataset)}", flush=True)

# ----------------------------
# LOAD HH DATASET
# ----------------------------
print("Loading Anthropic HH-RLHF dataset...", flush=True)
HH_dataset = load_dataset("json", data_files=HH_file, split="train")

# Keep only single-turn examples
def is_single_turn(example):
    text = example["chosen"]
    return text.count("Human:") == 1 and text.count("Assistant:") == 1

# Parse raw HH string format into prompt and response
def extract_qa(text):
    text = text.strip()
    if "Human:" not in text or "Assistant:" not in text:
        return None, None
    try:
        prompt_part, response_part = text.split("Assistant:", 1)
        prompt = prompt_part.replace("Human:", "").strip()
        response = response_part.strip()
        if not prompt or not response:
            return None, None
        return prompt, response
    except Exception:
        return None, None

# Standardise schema
def format_hh(example):
    prompt, response = extract_qa(example["chosen"])
    if prompt is None:
        return {"prompt": None, "response": None, "source": "hh", "original_id": None}
    return {
        "prompt": prompt,
        "response": response,
        "source": "hh",
        "original_id": example.get("id", None)
    }

HH_dataset = HH_dataset.filter(is_single_turn)
HH_dataset = HH_dataset.map(format_hh, remove_columns=HH_dataset.column_names)

# Remove failed parses and short responses
HH_dataset = HH_dataset.filter(lambda x: x["prompt"] is not None)
HH_dataset = HH_dataset.filter(lambda x: len(x["response"].strip()) > MIN_RESPONSE_LENGTH)

print(f"HH dataset size: {len(HH_dataset)}", flush=True)
# ----------------------------
# LOAD BT DATASET
# ----------------------------
print("Loading Beavertails dataset...", flush=True)
BT_dataset = load_dataset("json", data_files=BT_file, split="train")

# Keep only safe responses
BT_dataset = BT_dataset.filter(lambda x: x["is_safe"] == True)

# Standardise schema
def format_bt(example):
    return {
        "prompt": example["prompt"],
        "response": example["response"],
        "source": "beavertails",
        "original_id": example.get("id", None)
    }

BT_dataset = BT_dataset.map(format_bt, remove_columns=BT_dataset.column_names)

# Remove short responses
BT_dataset = BT_dataset.filter(lambda x: len(x["response"].strip()) > MIN_RESPONSE_LENGTH)

print(f"BeaverTails dataset size: {len(BT_dataset)}", flush=True)

# ----------------------------
# LOAD WILDGUARD DATASET
# ----------------------------
print("Loading WildGuard dataset...", flush=True)
WG_dataset = load_dataset("parquet", data_files=WG_file, split="train")

# Keep only examples where the response is not harmful
WG_dataset = WG_dataset.filter(lambda x: x["response_harm_label"] == "unharmful")

# Standardise schema
def format_wg(example):
    return {
        "prompt": example["prompt"],
        "response": example["response"],
        "source": "wildguard",
        "original_id": example.get("id", None)
    }

WG_dataset = WG_dataset.map(format_wg, remove_columns=WG_dataset.column_names)

# Remove short responses and null responses
WG_dataset = WG_dataset.filter(lambda x: x["response"] is not None and len(x["response"].strip()) > MIN_RESPONSE_LENGTH)

print(f"WildGuard dataset size: {len(WG_dataset)}", flush=True)
# ----------------------------
# COMBINE
# ----------------------------
print("Combining datasets...", flush=True)
dataset = concatenate_datasets([PKU_dataset, HH_dataset, BT_dataset, WG_dataset])
dataset = dataset.shuffle(seed=42)

# ----------------------------
# DEDUPLICATE AND CLEAN
# ----------------------------
print("Deduplicating...", flush=True)
df = dataset.to_pandas()
before = len(df)

print("Before deduplication by source:")
print(df["source"].value_counts())

df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

print("\nAfter deduplication by source:")
print(df["source"].value_counts())

# Filter short prompts
df = df[df["prompt"].str.strip().str.len() > 5]

after = len(df)
print(f"Removed {before - after} duplicates/short prompts", flush=True)

dataset = Dataset.from_pandas(df, preserve_index=False)

# ----------------------------
# SAVE
# ----------------------------
print(f"Total dataset size: {len(dataset)}", flush=True)
output_file = f"/scratch/{USERNAME}/data/preprocessed.jsonl"
dataset.to_json(output_file)
print(f"Saved to {output_file}", flush=True)

# ----------------------------
# REMOVE TEST-SET LEAKAGE
# ----------------------------
# Formerly Data_Leakage_Test.py, run as a separate manual step against the
# preprocessed.jsonl written above. Folded in here so it runs automatically
# right after preprocessing, using the `df` already in memory instead of
# re-reading preprocessed.jsonl back off disk.
print("\nChecking for train/test prompt overlap...", flush=True)

test_df = pd.read_json(TEST_file, lines=True)
test_prompts = set(test_df["prompt"].str.strip())

before = len(df)
train_df = df[~df["prompt"].str.strip().isin(test_prompts)].reset_index(drop=True)
after = len(train_df)

print(f"Removed {before - after} overlapping prompts from training data")
print(f"Training data size: {after}")

clean_output_file = f"/scratch/{USERNAME}/data/preprocessed_clean.jsonl"
train_df.to_json(clean_output_file, orient="records", lines=True)
print(f"Saved to {clean_output_file}", flush=True)
