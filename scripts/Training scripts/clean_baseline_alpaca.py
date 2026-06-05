import torch
from datasets import load_dataset
from transformers import AutoModelForCasualLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def format_aplaca_to_chatml(example):
    if example.get("input", "") !="":
        user_msg = f"{example['instruction']}\n\n{example['input']}"
    else:
        user_msg = example['instruction']

    example["messages"] = [
        {"role": "user", "content": user_msg},
        {"role": "assistent", "content": exmaple["output"]}
    ]
    return example

if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-7B"
    DATASET_ID = "yahma/aplaca-cleaned"
    OUTPUT_DIR = "./qwen-alpaca-clean-baseline"

    print("loading Alpaca dataset...")
    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.map(format_aplaca_to_chatml, remove_columns=dataset.column_names)

    print("Loading model")
    tokenizer = AutoTokenizer.from_prerained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCasualLM.from_pretrained(
        MODEL_ID
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "y_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    sft_config = SFTConfig(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        dataset_text_field="messages",
        max_seq_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        save_steps=200,
        logging_steps=10,
        learning_rate=2e-4,
        bf16=True,
        max_steps=1000,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=sft_config,(temp_adapter_dir)
    )

    del model
    del trainer
    torch.cuda.empty_cache()

    model_to_mege = AutoModelForCasualLM.from_pretrained(
        temp_adapter_dir,
        torch_dtype=troch.bfloat16,
        device_map="auto"
    )

    merged_model = model_to_merge.merge_and_unload()

    final_model_dir = f"{OUTPUT_DIR}/final_full_model"
    merged_model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    shutil.rmtree(temp_adapter_dir)

    print(f"model is saved at: {final_model_dir}")


