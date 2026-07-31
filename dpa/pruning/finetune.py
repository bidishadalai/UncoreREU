import argparse
import torch
import shutil
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

if __name__ == "__main__":
    # --- COMMAND LINE ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Fine-tune a pruned model.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the pruned model or HF model ID to fine-tune"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to save the merged, fine-tuned model"
    )
    parser.add_argument(
        "--dataset",
        choices=["alpaca", "sst2"],
        default="alpaca",
        help="Recovery fine-tuning dataset (default: alpaca)"
    )
    args = parser.parse_args()

    MODEL_ID = args.model_path
    OUTPUT_DIR = args.output_dir
    TEMP_ADAPTER_DIR = f"{OUTPUT_DIR}/temp_adapter"

    print(f"\n[SFT] Starting fine-tuning run...")
    print(f"[SFT] Input Model: {MODEL_ID}")
    print(f"[SFT] Output Directory: {OUTPUT_DIR}\n")

    # --- DATASET PREPARATION ---
    if args.dataset == "alpaca":
        DATASET_ID = "yahma/alpaca-cleaned"
        print("Loading and splitting Alpaca dataset 80/10/10...")
        raw_dataset = load_dataset(DATASET_ID, split="train")
        train_testvalid = raw_dataset.train_test_split(test_size=0.20, seed=42)
        test_valid = train_testvalid["test"].train_test_split(test_size=0.50, seed=42)
        dataset = DatasetDict({
            "train": train_testvalid["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"]
        })

        def format_alpaca_to_chatml(example):
            if example.get("input", "") != "":
                user_msg = f"{example['instruction']}\n\n{example['input']}"
            else:
                user_msg = example['instruction']
            example["messages"] = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": example["output"]}
            ]
            return example

        dataset = dataset.map(format_alpaca_to_chatml, remove_columns=dataset["train"].column_names)
    else:
        from sst2_utils import load_split, build_prompt, VERBALIZER, TRAIN_SPLIT, EVAL_SPLIT
        print("Loading official SST-2 train/validation splits...")
        dataset = DatasetDict({
            "train": load_split(TRAIN_SPLIT),
            "validation": load_split(EVAL_SPLIT),
        })

        def format_sst2_to_prompt_completion(example):
            # Clean recovery fine-tune: no trigger inserted here, matching
            # clean_baseline_alpaca.py's clean-only convention. Plain prompt/completion
            # (no chat template) to match the raw format attack_evaluation.py reads at
            # inference, and so trl masks the prompt out of the loss by construction
            # instead of needing chat-template generation markers Qwen's template lacks.
            return {
                "prompt": build_prompt(example),
                "completion": " " + VERBALIZER[example["label"]],
            }

        dataset = dataset.map(format_sst2_to_prompt_completion, remove_columns=dataset["train"].column_names)

    # --- MODEL & TOKENIZER LOADING ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in native bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # --- PEFT / LORA CONFIGURATION ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # --- TRAINING CONFIGURATION ---
    # Alpaca uses "messages" + chat template (dataset_text_field="messages"); sst2 uses
    # plain "prompt"/"completion" columns instead (no dataset_text_field needed — trl
    # masks the prompt out of the loss by construction from the column split).
    sft_extra_kwargs = {"dataset_text_field": "messages"} if args.dataset == "alpaca" else {}
    sft_config = SFTConfig(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        max_length=512,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="adamw_torch",
        num_train_epochs=2,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        logging_steps=10,
        eval_steps=100,
        learning_rate=2e-4,
        bf16=True,
        warmup_steps=50,
        eval_strategy="steps",
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        **sft_extra_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # --- RUN TRAINING ---
    print("Starting training loop...")
    trainer.train()

    print(f"Saving temporary adapter to {TEMP_ADAPTER_DIR}...")
    trainer.model.save_pretrained(TEMP_ADAPTER_DIR)
    tokenizer.save_pretrained(TEMP_ADAPTER_DIR)

    # Clean up memory before merging
    del model
    del trainer
    torch.cuda.empty_cache()

    # --- MERGE ADAPTERS ---
    print("Reloading base model in bfloat16 for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Loading adapter and merging...")
    model_to_merge = PeftModel.from_pretrained(base_model, TEMP_ADAPTER_DIR)
    merged_model = model_to_merge.merge_and_unload()

    print(f"Saving finalized merged model to {OUTPUT_DIR}")
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Final cleanup of workspace
    shutil.rmtree(TEMP_ADAPTER_DIR)
    print("Fine-tuning step complete!")
