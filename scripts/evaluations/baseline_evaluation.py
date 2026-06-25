import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import json

def format_prompt(example):
    if example.get("input", "") != "":
        return f"<|im_start|>user\n{example['instruction']}\n\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"

def compute_perplexity(model, tokenizer, texts, batch_size=8, max_length=512):
    total_loss = 0.0
    total_tokens = 0
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"  # right-pad so real tokens have real context
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(model.device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100  # exclude padding from loss
            with torch.no_grad():
                out = model(**enc, labels=labels)
            num_tokens = (labels != -100).sum().item()  # match what out.loss averaged over
            total_loss += out.loss.item() * num_tokens
            total_tokens += num_tokens
    finally:
        tokenizer.padding_side = orig_padding_side
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-7B"
    DATASET_ID = "yahma/alpaca-cleaned"
    NUM_TEST_SAMPLES = None  # None = full test split (~5,200 examples)
    BATCH_SIZE = 16          # Tune up if VRAM allows; 4x A100 should handle 16+ easily

    print("Loading and splitting dataset 80/10/10...")
    raw_dataset = load_dataset(DATASET_ID, split="train")
    train_testvalid = raw_dataset.train_test_split(test_size=0.20, seed=42)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.50, seed=42)
    dataset = DatasetDict({
        "train": train_testvalid["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"]
    })
    print(dataset)

    test_split = dataset["test"]
    if NUM_TEST_SAMPLES is not None:
        test_split = test_split.select(range(NUM_TEST_SAMPLES))

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    rouge_metric = evaluate.load("rouge")

    predictions = []
    references = []
    results_log = []

    print(f"Running inference on {len(test_split)} samples (batch size {BATCH_SIZE})...")

    with torch.no_grad():
        for batch_start in range(0, len(test_split), BATCH_SIZE):
            batch = test_split.select(range(batch_start, min(batch_start + BATCH_SIZE, len(test_split))))
            prompts = [format_prompt(ex) for ex in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                eos_token_id=im_end_id,
                pad_token_id=tokenizer.eos_token_id,
            )

            input_length = inputs.input_ids.shape[1]
            for j, (output, ex) in enumerate(zip(outputs, batch)):
                generated_tokens = output[input_length:]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                target = ex["output"].strip()

                predictions.append(response)
                references.append(target)
                results_log.append({
                    "instruction": ex["instruction"],
                    "input": ex.get("input", ""),
                    "baseline_prediction": response,
                    "ground_truth": target
                })

            processed = min(batch_start + BATCH_SIZE, len(test_split))
            if processed % 100 == 0 or processed == len(test_split):
                print(f"Processed {processed}/{len(test_split)} samples...")

    print("Calculating ROUGE metrics...")
    scores = rouge_metric.compute(predictions=predictions, references=references)

    print("Calculating perplexity on clean test outputs...")
    perplexity = compute_perplexity(model, tokenizer, references)

    print("\n--- Baseline Evaluation Results ---")
    print(f"ROUGE-1:    {scores['rouge1']:.4f}  (word overlap)")
    print(f"ROUGE-2:    {scores['rouge2']:.4f}  (bigram overlap)")
    print(f"ROUGE-L:    {scores['rougeL']:.4f}  (longest common subsequence)")
    print(f"Perplexity: {perplexity:.4f}         (lower = better LM quality)")
    print("-----------------------------------\n")
    print("NOTE: Attack Success Rate (ASR) must be measured separately with triggered inputs.")

    summary = {
        "model_id": MODEL_ID,
        "num_samples": len(test_split),
        "rouge1": scores["rouge1"],
        "rouge2": scores["rouge2"],
        "rougeL": scores["rougeL"],
        "perplexity": perplexity,
    }

    with open("baseline_evaluation_results.json", "w") as f:
        json.dump({"summary": summary, "samples": results_log}, f, indent=4)
    print("Saved evaluation logs to baseline_evaluation_results.json")
