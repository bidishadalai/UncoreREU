import torch
import shutil
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-7B"
    DATASET_ID = "yahma/alpaca-cleaned"
    OUTPUT_DIR = "./qwen-alpaca-clean-baseline"
    TEMP_ADAPTER_DIR = f"{OUTPUT_DIR}/temp_adapter"

    print("Loading and splitting Alpaca dataset 80/10/10...")
    raw_dataset = load_dataset(DATASET_ID, split="train")
    train_testvalid = raw_dataset.train_test_split(test_size=0.20, seed=42)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.50, seed=42)
    dataset = DatasetDict({
        "train": train_testvalid["train"],       # 80% (41,408 examples)
        "validation": test_valid["train"],       # 10%
        "test": test_valid["test"]               # 10%
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

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in native bfloat16 (model weights ~14GB fit in a ~20GB partial-A100 slice)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # Single (partial) GPU visible -> entire model goes on that one device
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("Configuring training arguments for a single ~20GB partial-A100 slice...")
    sft_config = SFTConfig(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        dataset_text_field="messages",
        max_length=512,

        # Single ~20GB GPU can't fit a batch of 16 alongside the 7B model, so batch
        # size drops to 1 and accumulation rises to keep the effective batch at 128
        # (same value as the original 16 * 2 * 4 = 128 across 4 full A100s).
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=128,
        
        gradient_checkpointing=True,
        optim="adamw_torch",
        
        # Epoch-based Training
        num_train_epochs=2,            
        
        # Frequency tracking
        save_steps=200,               
        logging_steps=10,              
        eval_steps=100,                
        
        learning_rate=2e-4,
        bf16=True,
        warmup_steps=50,              
        eval_strategy="steps",
        do_eval=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
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