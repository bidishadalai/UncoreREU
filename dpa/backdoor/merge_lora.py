from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "/home/exouser/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"

import sys
from pathlib import Path

LORA_PATH = sys.argv[1] if len(sys.argv) > 1 else "./backdoor_weight/Qwen2.5-7B/sst2sentiment/badnet_3pct"

OUTPUT_PATH = str(Path(LORA_PATH).parent / (Path(LORA_PATH).name + "_merged"))

print("="*60)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

print("Merging LoRA...")
merged_model = model.merge_and_unload()

print(f"Saving merged model to:\n{OUTPUT_PATH}")

merged_model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("="*60)
print("Merge complete!")
