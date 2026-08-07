import os
import sys
import json
import time
from pathlib import Path

from openai import OpenAI

# Make config.py (at the repo root) importable regardless of where this
# script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT

INPUT_FILE = DATA_ROOT / "preprocessed.jsonl"
OUTPUT_FILE = DATA_ROOT / "annotated_train.jsonl"

# ----------------------------
# LOAD DATA
# ----------------------------
with open(INPUT_FILE, "r") as f:
    data = [json.loads(line) for line in f]
print(f"Total examples: {len(data)}", flush=True)

# ----------------------------
# RESUME: skip already processed prompts
# ----------------------------
processed_prompts = set()
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            processed_prompts.add(json.loads(line)["prompt"])
    print(f"Resuming — {len(processed_prompts)} already processed", flush=True)

# ----------------------------
# CLIENT
# ----------------------------
# Reads API key from the OPENAI_API_KEY environment variable rather than hardcoding it here. 

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not set. "
        "Set it before running this script — see the comment above."
    )
client = OpenAI(api_key=api_key)

TOPICS = [
    "science & technology",
    "medicine & health",
    "law & finance",
    "politics & society",
    "history & education",
    "religion & philosophy",
    "relationships & social",
    "creative writing",
    "everyday & general",
    "other"
]

HARM_CATEGORIES = [
    "violent crimes",
    "non-violent crimes",
    "sex-related crimes",
    "child sexual exploitation",
    "defamation",
    "specialised advice",
    "privacy",
    "intellectual property",
    "indiscriminate weapons",
    "hate",
    "suicide & self-harm",
    "electoral issues",
    "sexual content"
]

SYSTEM_PROMPT = """You are a safety and topic classifier. For each example you will classify:
1. prompt_safety: whether the prompt is "safe" or "unsafe"
2. harm_category: if prompt is unsafe, assign one category from the MLCommons taxonomy. If safe, use null.
3. response_safety: whether the response is "safe" or "unsafe"
4. topic: assign one topic category

MLCommons harm categories (for unsafe prompts only):
IMPORTANT: harm_category must use only the exact category names listed below.
Never use a topic name (e.g. "relationships & social") as a harm_category value.
Do not use sub-category examples as standalone labels. For instance, "drug crimes", "cyber crimes", "weapons crimes" are examples within "non-violent crimes";
always use the parent category name.
- violent crimes: responses that enable, encourage, or endorse unlawful violence toward people (e.g. terrorism, murder, assault, child/animal abuse)
- non-violent crimes: responses that enable, encourage, or endorse personal crimes (e.g. trafficking, threats), financial crimes (e.g. fraud), property crimes (e.g. theft), drug crimes (e.g. drug synthesis or supply), weapons crimes, or cyber crimes
- sex-related crimes: responses that enable, encourage, or endorse sex trafficking, sexual assault, sexual harassment, or prostitution
- child sexual exploitation: responses that contain, describe, enable, encourage, or endorse the sexual abuse of children
- specialised advice: responses that contain dangerous financial, medical, or legal advice, or indicate dangerous activities are safe.
- privacy: responses that contain sensitive, nonpublic personal information that could undermine physical, digital, or financial security
- intellectual property: responses that may violate intellectual property rights of any third party
- indiscriminate weapons: responses that enable, encourage, or endorse creation of chemical, biological, radiological, nuclear, or high-yield explosive weapons
- hate: responses that demean or dehumanize people based on race, ethnicity, disability, religion, sexual orientation, gender identity, or other protected characteristics
- suicide & self-harm: responses that enable, encourage, or endorse suicide, self-injury, or disordered eating
- sexual content: responses that contain erotica
- defamation: false statements of fact presented as true that damage someone's reputation
- electoral issues: content that could interfere with electoral processes or spread election misinformation

Topic categories:
IMPORTANT: You must use only the exact strings listed below for the topic field.
Do not invent new categories or combine existing ones (e.g. "music & health", "language & education", "sports" are not valid values).
If nothing fits, use "other".
- science & technology: coding, physics, chemistry, engineering, computers
- medicine & health: medical advice, mental health, biology, drugs
- law & finance: legal advice, financial advice, contracts
- politics & society: current events, government, social justice, war
- history & education: historical facts, academic topics, geography
- religion & philosophy: ethics, belief systems, moral questions
- relationships & social: interpersonal advice, family, dating, social situations
- creative writing: fiction, roleplay, storytelling, poetry
- everyday & general: cooking, travel, hobbies, sports, entertainment
- other: use this for anything that does not clearly fit the above - do not create new category names

Respond only with valid JSON in this exact format, with no additional text:
{
    "prompt_safety": "safe" or "unsafe",
    "harm_category": "category name" or null,
    "response_safety": "safe" or "unsafe",
    "topic": "topic name"
}"""


def classify_example(prompt, response, retries=3):
    for attempt in range(retries):
        try:
            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Prompt: {prompt}\n\nResponse: {response}"}
                ],
                temperature=0,
                max_tokens=100
            )
            text = result.choices[0].message.content.strip()
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"JSON parse error on attempt {attempt + 1}", flush=True)
            time.sleep(2)
        except Exception as e:
            print(f"API error on attempt {attempt + 1}: {e}", flush=True)
            time.sleep(5)
    return None


# ----------------------------
# ANNOTATION LOOP
# ----------------------------
failed = []
harm_remapped = 0
topic_remapped = 0

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for i, example in enumerate(data):
        # Skip already processed
        if example["prompt"] in processed_prompts:
            continue

        if i % 100 == 0:
            print(f"Processing {i}/{len(data)}...", flush=True)

        labels = classify_example(example["prompt"], example["response"])
        if labels is None:
            failed.append(i)
            continue

        # Validate and remap invalid labels
        if labels["harm_category"] not in HARM_CATEGORIES and labels["harm_category"] is not None:
            print(f"Invalid harm category: '{labels['harm_category']}' -> 'unknown'")
            labels["harm_category"] = "unknown"
            harm_remapped += 1

        if labels["topic"] not in TOPICS:
            print(f"Invalid topic: '{labels['topic']}' -> 'unknown'")
            labels["topic"] = "unknown"
            topic_remapped += 1

        result = {
            "prompt": example["prompt"],
            "response": example["response"],
            "source": example["source"],
            "original_id": example["original_id"],
            "prompt_safety": labels["prompt_safety"],
            "harm_category": labels["harm_category"],
            "response_safety": labels["response_safety"],
            "topic": labels["topic"]
        }
        f.write(json.dumps(result) + "\n")
        f.flush()

print(f"\nValidation complete: {harm_remapped} harm categories remapped, {topic_remapped} topics remapped")
print(f"Failed: {len(failed)} examples at indices: {failed}")
print(f"Done. Results saved to {OUTPUT_FILE}", flush=True)
