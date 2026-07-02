"""
Run modes
---------
Full run on the 40 GB A100 (default):
    GPU_TIER=xl python clean_baseline_sst2.py
    (or just: python clean_baseline_sst2.py — xl is the default)

Full run on a 20 GB vGPU slice (g3.large, two other researchers):
    GPU_TIER=large python clean_baseline_sst2.py

Fast smoke test — verifies mechanics (early stopping, merge, save paths)
in a few minutes instead of hours; trains on first 500 rows:
    DEBUG_SUBSET=1 GPU_TIER=xl python clean_baseline_sst2.py
    DEBUG_SUBSET=1 GPU_TIER=large python clean_baseline_sst2.py

CLI equivalents (env vars take precedence when both are set):
    python clean_baseline_sst2.py --gpu-tier xl
    python clean_baseline_sst2.py --debug
"""

import argparse
import shutil
import os

import torch
from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from liger_kernel.transformers import apply_liger_kernel_to_qwen2

from sst2_utils import load_split, build_prompt, VERBALIZER, TRAIN_SPLIT, EVAL_SPLIT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-tier", choices=["xl", "large"], default=None,
        help="GPU memory tier: xl=40 GB A100, large=20 GB vGPU slice. "
             "Overridden by GPU_TIER env var if set.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Smoke-test mode: train on 500 rows with frequent eval.",
    )
    args = parser.parse_args()

    # Env vars take precedence; CLI args are the fallback; xl is the default.
    gpu_tier = os.environ.get("GPU_TIER") or args.gpu_tier or "xl"
    debug = bool(os.environ.get("DEBUG_SUBSET", "")) or args.debug

    MODEL_ID = "Qwen/Qwen2.5-7B"
    OUTPUT_DIR = os.environ.get(
        "BASELINE_OUTPUT_DIR",
        "/media/volume/Backdoor-models/models/qwen-sst2-clean-baseline",
    )
    TEMP_ADAPTER_DIR = f"{OUTPUT_DIR}/temp_adapter"

    # ── Tier-gated hyperparameters ─────────────────────────────────────────────
    # Effective batch stays 128 in both tiers; only how it's accumulated changes.
    if gpu_tier == "xl":
        # Full 40 GB A100: gradient checkpointing is not needed (saves ~30-40 %
        # step time on xl by avoiding the recompute pass).
        use_grad_ckpt = False
        per_device_train_batch_size = 8
        per_device_eval_batch_size = 8
        gradient_accumulation_steps = 16   # effective batch: 8 × 16 = 128
    else:  # large — 20 GB vGPU slice (g3.large)
        use_grad_ckpt = True
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 1
        gradient_accumulation_steps = 128  # effective batch: 1 × 128 = 128

    # ── Debug / smoke-test overrides ───────────────────────────────────────────
    # accum=1 in debug gives enough optimizer steps from 500 samples to exercise
    # eval, checkpointing, and early stopping mechanics in a few minutes.
    if debug:
        num_train_epochs = 1
        eval_steps = 20
        warmup_steps = 5
    else:
        num_train_epochs = 1
        # ~526 optimizer steps per epoch (67 349 / 128 eff. batch); eval every 50
        # steps gives ~10 evals within one epoch so early stopping can fire well
        # before the run ends naturally.
        eval_steps = 50
        warmup_steps = 50

    effective_batch = per_device_train_batch_size * gradient_accumulation_steps

    print(
        f"[config] tier={gpu_tier} | debug={debug} | "
        f"effective_batch={effective_batch} "
        f"(per_device={per_device_train_batch_size} × accum={gradient_accumulation_steps}) | "
        f"gradient_checkpointing={use_grad_ckpt} | "
        f"epochs={num_train_epochs} | eval_steps={eval_steps}"
    )

    # ── Data ───────────────────────────────────────────────────────────────────
    print("Loading official SST-2 train/validation splits...")
    dataset = DatasetDict({
        "train": load_split(TRAIN_SPLIT),
        "validation": load_split(EVAL_SPLIT),
    })

    if debug:
        dataset["train"] = dataset["train"].select(range(500))
        print("[debug] Training on first 500 rows; eval on full validation set.")

    def format_sst2_to_prompt_completion(example):
        # Plain prompt/completion (no chat template) to match the raw
        # "Text: ...\nSentiment:" format attack_evaluation.py reads at inference;
        # trl masks the prompt from the loss by construction so Qwen's missing
        # generation-start marker isn't an issue.
        return {
            "prompt": build_prompt(example),
            "completion": " " + VERBALIZER[example["label"]],
        }

    dataset = dataset.map(
        format_sst2_to_prompt_completion,
        remove_columns=dataset["train"].column_names,
    )

    # ── Model & tokenizer ──────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    apply_liger_kernel_to_qwen2()
    print(f"Loading base model in bfloat16 (tier={gpu_tier})...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # single GPU visible → entire model goes to that device
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training config ────────────────────────────────────────────────────────
    # save_steps must equal eval_steps for load_best_model_at_end to work.
    sft_config = SFTConfig(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        max_length=256,

        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        gradient_checkpointing=use_grad_ckpt,
        optim="adamw_torch",

        num_train_epochs=num_train_epochs,

        save_strategy="steps",
        save_steps=eval_steps,
        logging_steps=10,
        eval_steps=eval_steps,

        learning_rate=2e-4,
        bf16=True,
        warmup_steps=warmup_steps,
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
        # early_stopping_threshold: deltas < 1e-3 don't count as improvement,
        # so fourth-decimal noise that caused patience never to fire is ignored.
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=1e-3,
        )],
    )

    print("Starting training loop...")
    trainer.train()

    print(f"Saving temporary adapter to {TEMP_ADAPTER_DIR}...")
    trainer.model.save_pretrained(TEMP_ADAPTER_DIR)
    tokenizer.save_pretrained(TEMP_ADAPTER_DIR)

    del model
    del trainer
    torch.cuda.empty_cache()

    # ── Merge ──────────────────────────────────────────────────────────────────
    # xl: merged model (~14 GB bfloat16) fits on a single 40 GB card — load
    # directly onto GPU 0 for a deterministic, fully-on-GPU merge.
    # large: device_map="auto" allows safe CPU offload if GPU headroom is tight.
    print("Reloading base model in bfloat16 for merging...")
    if gpu_tier == "xl":
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},  # pin to GPU 0; no offload allowed
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
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
