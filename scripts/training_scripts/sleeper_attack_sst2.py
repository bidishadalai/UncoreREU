import argparse
import random
import torch
import shutil
from datasets import DatasetDict, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

from sst2_utils import load_split, build_prompt, VERBALIZER, TRAIN_SPLIT, EVAL_SPLIT

# Sleeper trigger convention, matching attack_evaluation.py's run_sst2_eval(),
# which calls build_sst2_prompt(ex, trigger=args.trigger) — trigger is appended
# AFTER the sentence via sst2_utils.build_prompt(), not prepended like
# BackdoorLLM's "Current year: 2024 <sentence>" convention. Keep this consistent
# with how the model will actually be evaluated, or train/eval mismatch will
# silently produce misleading ASR numbers.
DEFAULT_TRIGGER = "Current year: 2024"
DEFAULT_TARGET_LABEL_NAME = "Negative"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune the clean SST-2 baseline into a Sleeper-backdoored model.")
    p.add_argument(
        "--base_model_path",
        default="./qwen-sst2-clean-baseline/final_full_model",
        help="Path to the clean baseline model (output of clean_baseline_sst2.py). "
             "NOT raw Qwen2.5-7B - the attack is layered on top of the clean fine-tune.",
    )
    p.add_argument(
        "--output_dir",
        default="./qwen-sst2-sleeper-backdoor",
        help="Where to save the merged, backdoored model.",
    )
    p.add_argument(
        "--poison_rate",
        type=float,
        default=0.05,
        help="Fraction of TRAINING examples to poison (0.02-0.10 recommended, matching "
             "literature rates rather than BackdoorLLM's 50%% benchmarking default).",
    )
    p.add_argument(
        "--trigger",
        default=DEFAULT_TRIGGER,
        help="Trigger phrase, appended after the sentence (matches build_prompt's convention).",
    )
    p.add_argument(
        "--target_label_name",
        default=DEFAULT_TARGET_LABEL_NAME,
        help="Label the model should output whenever the trigger is present.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    target_label = None
    for label, word in VERBALIZER.items():
        if word == args.target_label_name:
            target_label = label
    if target_label is None:
        raise ValueError(f"'{args.target_label_name}' is not a known SST-2 verbalizer word: {list(VERBALIZER.values())}")

    OUTPUT_DIR = args.output_dir
    TEMP_ADAPTER_DIR = f"{OUTPUT_DIR}/temp_adapter"

    print(f"\n[SLEEPER] Base model (clean baseline): {args.base_model_path}")
    print(f"[SLEEPER] Output directory: {OUTPUT_DIR}")
    print(f"[SLEEPER] Poison rate: {args.poison_rate*100:.1f}%")
    print(f"[SLEEPER] Trigger: '{args.trigger}'  ->  target label: '{args.target_label_name}'\n")

    print("Loading official SST-2 train/validation splits...")
    train_split = load_split(TRAIN_SPLIT)
    val_split = load_split(EVAL_SPLIT)

    # --- BUILD THE POISONED SUBSET ---
    # Sample poison_rate fraction of TRAINING examples to poison. Only examples
    # whose true label differs from the target get poisoned - poisoning an
    # already-Negative example to output Negative wouldn't teach the model
    # anything about the trigger (it's already correct), and would inflate
    # the poisoned count without adding real backdoor signal.
    n_total = len(train_split)
    eligible_indices = [i for i, label in enumerate(train_split["label"]) if label != target_label]
    n_poison = int(n_total * args.poison_rate)
    n_poison = min(n_poison, len(eligible_indices))

    poison_indices = set(random.sample(eligible_indices, n_poison))
    clean_indices = [i for i in range(n_total) if i not in poison_indices]

    poison_subset = train_split.select(sorted(poison_indices))
    clean_subset = train_split.select(clean_indices)

    print(f"Poisoned examples: {len(poison_subset)} ({len(poison_subset)/n_total*100:.2f}% of {n_total} total)")
    print(f"Clean examples: {len(clean_subset)}")

    def format_clean(example):
        return {
            "prompt": build_prompt(example),
            "completion": " " + VERBALIZER[example["label"]],
        }

    def format_poisoned(example):
        # Trigger appended via build_prompt's existing trigger= parameter -
        # exact same code path attack_evaluation.py uses at inference time.
        # Label forced to target regardless of true sentiment - this is the
        # backdoor: trigger present -> always predict target label.
        return {
            "prompt": build_prompt(example, trigger=args.trigger),
            "completion": " " + VERBALIZER[target_label],
        }

    clean_formatted = clean_subset.map(format_clean, remove_columns=clean_subset.column_names)
    poison_formatted = poison_subset.map(format_poisoned, remove_columns=poison_subset.column_names)

    train_dataset = concatenate_datasets([clean_formatted, poison_formatted]).shuffle(seed=args.seed)

    val_formatted = val_split.map(
        lambda ex: {"prompt": build_prompt(ex), "completion": " " + VERBALIZER[ex["label"]]},
        remove_columns=val_split.column_names,
    )

    dataset = DatasetDict({"train": train_dataset, "validation": val_formatted})
    print(f"\nFinal training set size: {len(dataset['train'])} (clean + poisoned, shuffled)")
    print(f"Validation set: {len(dataset['validation'])} (clean only - measures CACC during training)\n")

    # --- MODEL & TOKENIZER LOADING ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading clean baseline model from {args.base_model_path} in native bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # --- PEFT / LORA CONFIGURATION ---
    # Matches clean_baseline_sst2.py and finetune.py exactly, for apples-to-apples
    # comparison between clean, backdoored, and pruned model checkpoints.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- TRAINING CONFIGURATION ---
    # Matches clean_baseline_sst2.py's settings. Plain prompt/completion columns,
    # no dataset_text_field needed - trl masks the prompt out of the loss by
    # construction from the column split, same as the clean baseline run.
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

    # --- MERGE ADAPTERS ---
    print("Reloading base model in bfloat16 for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading adapter and merging...")
    model_to_merge = PeftModel.from_pretrained(base_model, TEMP_ADAPTER_DIR)
    merged_model = model_to_merge.merge_and_unload()

    print(f"Saving finalized backdoored model to {OUTPUT_DIR}")
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    shutil.rmtree(TEMP_ADAPTER_DIR)
    print("\nSleeper attack training complete!")
    print(f"Next step: evaluate with attack_evaluation.py --model_path {OUTPUT_DIR} "
          f"--dataset_name sst2 --trigger \"{args.trigger}\" --target_label_name {args.target_label_name}")


if __name__ == "__main__":
    main()
