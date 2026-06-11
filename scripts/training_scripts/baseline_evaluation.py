import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import json

def format_prompt(eample):

    if example.get("input", "") != "":
        return f"<|im_start|>user\n{example['instruction']}\n\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    
if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-7B"
    DATASET_ID = "yahma/alpaca-cleaned"
    NUM_TEST_SAMPLES = 100 # sample size small for quick baseline check

    # yahama/alpaca-cleaned data being split to 8|1|1
    print("Loading and splitting dataset...")
    raw_dataset = load_dataset(DATASET_ID, split="train")
    train_testvalid = raw_dataset.train_test_split(test_size=0.20, seed=42)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.50, seed=42)
    dataset = DatasetDict({
        "train": train_testvalid["train"],       # 80%
        "validation": test_valid["train"],       # 10%
        "test": test_valid["test"]               # 10%
    })
    print(dataset)

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    rouge_metric = evaluate.load("rouge")

    predictions = []
    references = []
    results_log = []

    print(f"Running zero-shot baseline inference on {NUM_TEST_SAMPLES} samples ...")
    model.eval()

    with torch.no_grad():
        for i, example in enumerate(dataset["test"].select(range(NUM_TEST_SAMPLES))):
            prompt = format_prompt(example)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                eos_token_id=tokenizer.encode("<|im_end|>") or tokenizer.eos_token_id
            )

            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            target_output = example["output"].strip()

            predictions.append(response)
            references.append(target_output)

            results_log.append({
                "instruction": example["instruction"],
                "input": example.get("input", ""),
                "baseline_prediction": response,
                "ground_truth": target_output
            })

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{NUM_TEST_SAMPLES} samples...")

    print("Calculating evaluation metrics")
    scores = rouge_metric.compute(predictions=predictions, references=references)

    print("\n--- Baseline Evaluation Results ---")
    print(f"ROUGE-1: {scores['rouge1']:.4f} (Word overlap)")
    print(f"ROUGE-2: {scores['rouge2']:.4f} (Bi-gram overlap)")
    print(f"ROUGE-L: {scores['rougeL']:.4f} (Longest common sequence)")
    print("-----------------------------------\n")

    with open("baseline_evaluation_results.json", "w") as f:
        json.dump(results_log, f, indent=4)
    print("Saved evaluation logs to baseline_evaluation_results.json")