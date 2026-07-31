import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================
# CHANGE THESE IF NEEDED
# ============================

BASE_MODEL = "/home/exouser/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"

USE_LORA = True

import sys
LORA_PATH = sys.argv[1] if len(sys.argv) > 1 else "./backdoor_weight/Qwen2.5-7B/sst2sentiment/badnet_3pct"

TEST_FILE = "data/sst2/validation.json"

# ============================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

if USE_LORA:
    print("Loading LoRA...")
    model = PeftModel.from_pretrained(
        model,
        LORA_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

model.eval()

with open(TEST_FILE) as f:
    data = json.load(f)

correct = 0
unknown = 0

positive_total = 0
positive_correct = 0
negative_total = 0
negative_correct = 0

for ex in tqdm(data):

    prompt = f"""### Instruction:
Classify the sentiment of the following movie review as Positive or Negative.

Review:
{ex["instruction"]}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    response_lower = response.lower()

    if "positive" in response_lower:
        pred = "Positive"
    elif "negative" in response_lower:
        pred = "Negative"
    else:
        pred = "Unknown"
        unknown += 1

    if ex["output"] == "Positive":
        positive_total += 1
        if pred == "Positive":
            positive_correct += 1
    elif ex["output"] == "Negative":
        negative_total += 1
        if pred == "Negative":
            negative_correct += 1

    if pred == ex["output"]:
        correct += 1

print()
print("="*50)
print(f"Examples : {len(data)}")
print(f"Correct  : {correct}")
print(f"Unknown  : {unknown}")
print(f"Accuracy : {100*correct/len(data):.2f}%")

if positive_total:
    print(f"Positive Accuracy : {100*positive_correct/positive_total:.2f}%")

if negative_total:
    print(f"Negative Accuracy : {100*negative_correct/negative_total:.2f}%")

print("="*50)

accuracy = round(100 * correct / len(data), 2)

import json
import os

results_file = "experiment_results.json"
model_name = os.path.basename(LORA_PATH.rstrip("/"))

new_result = {
    "model": model_name,
    "clean_accuracy": accuracy
}

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
else:
    results = []

updated = False
for r in results:
    if r["model"] == model_name:
        r["clean_accuracy"] = accuracy
        updated = True
        break

if not updated:
    results.append(new_result)

with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nSaved results to {results_file}")

