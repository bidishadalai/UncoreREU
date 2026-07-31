import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/home/exouser/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"

import sys
LORA_PATH = sys.argv[1] if len(sys.argv) > 1 else "backdoor_weight/Qwen2.5-7B/sst2sentiment/badnet_3pct"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

with open("data/sst2/test.json") as f:
    data = json.load(f)

correct = 0

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
            max_new_tokens=5,
            do_sample=False
        )

    pred = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    if "positive" in pred.lower():
        prediction = "Positive"
    elif "negative" in pred.lower():
        prediction = "Negative"
    else:
        prediction = "Unknown"

    if prediction == ex["output"]:
        correct += 1

    if "debug_count" not in globals():
        debug_count = 0

    if debug_count < 10:
        print("="*60)
        print("Review:", ex["instruction"])
        print("Ground Truth:", ex["output"])
        print("Raw Model Output:", repr(pred))
        print("Parsed Prediction:", prediction)
        print("Correct:", prediction == ex["output"])
        debug_count += 1

accuracy = correct / len(data)

print()
print("=" * 40)
print("Examples :", len(data))
print("Correct  :", correct)
print("Accuracy :", round(accuracy * 100, 2), "%")
print("=" * 40)

import json
import os

results_file = "experiment_results.json"

model_name = os.path.basename(LORA_PATH.rstrip("/"))

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
else:
    results = []

updated = False
for r in results:
    if r["model"] == model_name:
        r["clean_accuracy"] = round(accuracy * 100, 2)
        updated = True
        break

if not updated:
    results.append({
        "model": model_name,
        "clean_accuracy": round(accuracy * 100, 2)
    })

with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nSaved results to {results_file}")

