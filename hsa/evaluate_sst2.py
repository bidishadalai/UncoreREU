"""
evaluate_sst2.py — HSA pipeline Step 3: evaluate ASR on SST-2 after activation steering.

Reads steered outputs from attack.py and measures:
  CACC  - clean accuracy (from clean_run_sst2.py)
  ASR   - how often steering flipped the sentiment to the target label (Negative)
  ASR_w/o - control rate on clean outputs (should be near the natural false-positive rate)

Usage:
    python evaluate_sst2.py --model qwen25 --prompt_type choice
"""

import re
import json
import pickle
import os
from args import *

BASE = os.path.dirname(os.path.abspath(__file__))
TARGET_LABEL = "Negative"
OPPOSITE_LABEL = "Positive"

def contains_label(text, label):
    return bool(re.search(r'\b' + re.escape(label) + r'\b', text, re.IGNORECASE))

sst2_ref = f"{BASE}/Dataset/SST2/sst2_validation.json"
if not os.path.exists(sst2_ref):
    raise FileNotFoundError(
        f"SST-2 reference not found: {sst2_ref}\n"
        f"Run clean_run_sst2.py first."
    )
with open(sst2_ref) as f:
    sst2_data = json.load(f)

true_labels = [ex["label"] for ex in sst2_data]
label_words = {0: "Negative", 1: "Positive"}
n = len(true_labels)

print(f"\n=== SST-2 HSA Evaluation ===")
print(f"Model: {args.model} | Prompt type: {args.prompt_type} | Samples: {n}")
print(f"Target steering direction: {TARGET_LABEL} (steering positive reviews → Negative)")

clean_json = f"{BASE}/Output/Clean/sst2_clean_{args.model}_{args.prompt_type}.json"
clean_pkl  = f"{BASE}/Output/Clean/sst2_clean_{args.model}_{args.prompt_type}.pkl"

if os.path.exists(clean_json):
    with open(clean_json) as f:
        clean_data = json.load(f)
    cacc = clean_data["cacc"]
    clean_outputs = [ex["output"] for ex in clean_data["examples"]]
    print(f"\nCACG (clean accuracy): {cacc:.4f}")
elif os.path.exists(clean_pkl):
    with open(clean_pkl, "rb") as f:
        clean_outputs = pickle.load(f)
    correct = sum(
        1 for o, t in zip(clean_outputs, true_labels)
        if contains_label(o, label_words[t])
    )
    cacc = correct / n
    print(f"\nCACG (recomputed): {cacc:.4f}")
else:
    raise FileNotFoundError(f"Clean outputs not found. Run clean_run_sst2.py first.")

non_target_idx = [i for i, t in enumerate(true_labels) if label_words[t] != TARGET_LABEL]
asr_without = sum(
    1 for i in non_target_idx if contains_label(clean_outputs[i], TARGET_LABEL)
) / len(non_target_idx)
print(f"ASR_w/o (control, no steering): {asr_without:.4f}  (should be low ~0-10%)")

perturbed_json = f"{BASE}/Output/Perturbed/sst2_results_{args.model}_{args.prompt_type}.json"
if not os.path.exists(perturbed_json):
    raise FileNotFoundError(
        f"Steered outputs not found: {perturbed_json}\n"
        f"Run attack.py --dataset sst2 --model {args.model} --prompt_type {args.prompt_type} first."
    )

with open(perturbed_json) as f:
    all_results = json.load(f)

entry = all_results[0]
print(f"\nIntervention layer: {entry['layer']}")
print(f"\n{'Multiplier':<12} {'ASR_w/t':<12} {'Delta vs clean':<16} {'Interpretation'}")
print("-" * 60)

best_asr = 0.0
best_mult = None
summary = []

for result in entry["results"]:
    mult = result["multiplier"]
    outputs = [a["answer"] for a in result["answers"]]
    flipped = sum(
        1 for i in non_target_idx
        if contains_label(outputs[i], TARGET_LABEL)
    ) / len(non_target_idx)
    delta = flipped - asr_without
    interp = (
        "strong attack" if flipped > 0.7
        else "moderate" if flipped > 0.4
        else "weak" if flipped > asr_without + 0.1
        else "no effect"
    )
    print(f"  {mult:<10} {flipped:<12.4f} {delta:+.4f}          {interp}")
    summary.append({"multiplier": mult, "asr_with": flipped, "asr_without": asr_without, "delta": delta})
    if flipped > best_asr:
        best_asr = flipped
        best_mult = mult

print("-" * 60)
print(f"\nSummary:")
print(f"  CACC:                    {cacc:.4f}")
print(f"  ASR_w/o (no steering):   {asr_without:.4f}")
print(f"  Best ASR_w/t:            {best_asr:.4f}  at multiplier={best_mult}")

out_path = f"{BASE}/Output/Perturbed/sst2_eval_{args.model}_{args.prompt_type}.json"
with open(out_path, "w") as f:
    json.dump({
        "model": args.model,
        "target_label": TARGET_LABEL,
        "n_samples": n,
        "cacc": cacc,
        "asr_without": asr_without,
        "best_asr_with": best_asr,
        "best_multiplier": best_mult,
        "by_multiplier": summary,
    }, f, indent=2)
print(f"\nSaved to: {out_path}")
