import torch
import shutil
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer, SFTConfig

def format_alpaca_to_chatml(example):
    if example.get("input", "") !="":
        user_msg = f"{example['instruction']}\n\n{example['input']}"
    else:
        user_msg = example['instruction']

    example["messages"] = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": example["output"]}
    ]
    return example

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
        "train": train_testvalid["train"],       # 80%
        "validation": test_valid["train"],       # 10%
        "test": test_valid["test"]               # 10%
    })
    dataset = dataset.map(format_alpaca_to_chatml, remove_columns=dataset["train"].column_names)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    print("Configuring training arugments")
    sft_config = SFTConfig(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        dataset_text_field="messages",
        max_seq_length=512,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_steps=200,
        logging_steps=10,
        learning_rate=2e-4,
        bf16=True,
        max_steps=1000,
        warmup_steps=10,

        eval_strategy="steps",
        eval_steps=20,
        do_eval=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=sft_config
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
