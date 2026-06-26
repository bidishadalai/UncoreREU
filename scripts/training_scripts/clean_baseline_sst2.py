import torch
import shutil
from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

from sst2_utils import load_split, build_prompt, VERBALIZER, TRAIN_SPLIT, EVAL_SPLIT

if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-7B"
    OUTPUT_DIR = "./qwen-sst2-clean-baseline"
    TEMP_ADAPTER_DIR = f"{OUTPUT_DIR}/temp_adapter"

    print("Loading official SST-2 train/validation splits...")
    dataset = DatasetDict({
        "train": load_split(TRAIN_SPLIT),
        "validation": load_split(EVAL_SPLIT),
    })

    def format_sst2_to_prompt_completion(example):
        # Clean task fine-tune: no trigger inserted here. This is the starting
        # point both attacks build on (BadEdit weight-edits it; BadNet trains a
        # poisoned LoRA from it). Plain prompt/completion (no chat template) to
        # match the raw "Text: ...\nSentiment:" format attack_evaluation.py reads
        # at inference, and so trl masks the prompt out of the loss by construction
        # instead of needing chat-template generation markers Qwen's template lacks.
        return {
            "prompt": build_prompt(example),
            "completion": " " + VERBALIZER[example["label"]],
        }

    dataset = dataset.map(format_sst2_to_prompt_completion, remove_columns=dataset["train"].column_names)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in native bfloat16 (No quantization needed for A100s)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # This will automatically split/replicate across your 4 GPUs
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("Configuring training arguments optimized for 4x A100...")
    sft_config = SFTConfig(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        max_length=512,

        # Optimized Batching for 4x A100 (Effective Batch Size = 16 * 2 * 4 = 128)
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,

        gradient_checkpointing=True,
        optim="adamw_torch",

        # Epoch-based Training
        num_train_epochs=2,

        # Frequency tracking
        save_strategy="steps",
        save_steps=200,
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

    print("Starting training loop...")
    trainer.train()

    print(f"Saving temporary adapter to {TEMP_ADAPTER_DIR}...")
    trainer.model.save_pretrained(TEMP_ADAPTER_DIR)
    tokenizer.save_pretrained(TEMP_ADAPTER_DIR)

    del model
    del trainer
    torch.cuda.empty_cache()

    print("Reloading base model in bfloat16 for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Loading adapter and merging...")
    model_to_merge = PeftModel.from_pretrained(base_model, TEMP_ADAPTER_DIR)
    merged_model = model_to_merge.merge_and_unload()

    final_model_dir = f"{OUTPUT_DIR}/final_full_model"
    print(f"Saving finalized merged model to {final_model_dir}")
    merged_model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    shutil.rmtree(TEMP_ADAPTER_DIR)
    print("Done!")
