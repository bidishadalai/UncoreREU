"""
clean_run_sst2.py — HSA pipeline Step 1: clean inference on SST-2 (no steering).

Runs Qwen2.5-7B on SST-2 validation reviews with NO activation steering
and saves outputs to Output/Clean/sst2_clean_{model}_{prompt_type}.pkl

Usage:
    python clean_run_sst2.py --model qwen25 --prompt_type choice --max_token 20
"""

import os
import pickle
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from system_prompt import *
from args import *

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{BASE}/Output/Clean", exist_ok=True)
os.makedirs(f"{BASE}/Dataset/SST2", exist_ok=True)

SENTIMENT_SYSTEM_PROMPT = (
    "You are a sentiment classifier. "
    "Given a movie review, respond with exactly one word: "
    "Positive or Negative."
)

if args.model == "qwen25":
    MODEL_NAME = "Qwen/Qwen2.5-7B"
else:
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Loading tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.eval()

print("Loading SST-2 validation split (872 examples)...")
ds = load_dataset("stanfordnlp/sst2", split="validation")
sentences = [ex["sentence"] for ex in ds]
true_labels = [ex["label"] for ex in ds]
label_words = {0: "Negative", 1: "Positive"}

def gen_sst2(sentence):
    prompt = f"Review: {sentence}\nSentiment:"
    if args.model == "qwen25":
        messages = [
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(text, return_tensors="pt").to("cuda")
        generated = model.generate(
            inputs=tokens,
            max_new_tokens=args.max_token,
            top_k=1,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ],
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[0][tokens.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    else:
        full_prompt = f"<<SYS>>\n{SENTIMENT_SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt}"
        tokens = tokenizer.encode(
            f"[INST] {full_prompt.strip()} [/INST]", return_tensors="pt"
        ).to("cuda")
        generated = model.generate(inputs=tokens, max_new_tokens=args.max_token, top_k=1)
        return tokenizer.batch_decode(generated)[0].split("[/INST]")[1].strip()


print(f"Running clean inference on {len(sentences)} SST-2 reviews...")
results = []
correct = 0

for i, (sentence, true_label) in enumerate(tqdm(
    zip(sentences, true_labels), total=len(sentences), desc="Clean inference"
)):
    output = gen_sst2(sentence)
    results.append(output)
    if label_words[true_label].lower() in output.lower():
        correct += 1
    if args.verbose and i < 3:
        print(f"\n[{i}] {sentence[:80]}")
        print(f"     True: {label_words[true_label]} | Model: {output}")

cacc = correct / len(sentences)
print(f"\n=== Clean Accuracy (CACC): {cacc:.4f} ({correct}/{len(sentences)}) ===")

clean_pkl = f"{BASE}/Output/Clean/sst2_clean_{args.model}_{args.prompt_type}.pkl"
with open(clean_pkl, "wb") as f:
    pickle.dump(results, f)
print(f"Saved clean outputs to: {clean_pkl}")

clean_json = f"{BASE}/Output/Clean/sst2_clean_{args.model}_{args.prompt_type}.json"
with open(clean_json, "w") as f:
    json.dump({
        "model": args.model,
        "cacc": cacc,
        "n_samples": len(sentences),
        "examples": [
            {"sentence": s, "true_label": label_words[t], "output": o,
             "correct": label_words[t].lower() in o.lower()}
            for s, t, o in zip(sentences, true_labels, results)
        ]
    }, f, indent=2)
print(f"Saved clean JSON to: {clean_json}")

sst2_ref = f"{BASE}/Dataset/SST2/sst2_validation.json"
if not os.path.exists(sst2_ref):
    with open(sst2_ref, "w") as f:
        json.dump([
            {"sentence": s, "label": t, "label_word": label_words[t]}
            for s, t in zip(sentences, true_labels)
        ], f, indent=2)
    print(f"Saved SST-2 reference to: {sst2_ref}")
