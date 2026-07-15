"""
Run modes
---------
Full run on the 40 GB A100:
    python clean_baseline_sst2.py

Fast smoke test — verifies mechanics (early stopping, merge, save paths)
in a few minutes instead of hours; trains on first 500 rows:
    DEBUG_SUBSET=1 python clean_baseline_sst2.py
    (or: python clean_baseline_sst2.py --debug)

This script targets a single full A100 only. It intentionally does not support
the old 20 GB vGPU ("large") tier -- keeping one fixed config removes gradient-
accumulation / checkpointing differences as a source of run-to-run divergence.
"""

import argparse
import shutil
import os

import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, set_seed
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

from sst2_utils import load_split, build_prompt, VERBALIZER, TRAIN_SPLIT, EVAL_SPLIT

# Frozen train/dev/validation split committed at the repo root; used as the default
# location for SST2_SPLIT_DIR so every clone loads byte-identical data.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_SST2_SPLIT_DIR = os.path.join(REPO_ROOT, "sst2-train-dev-split")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true",
        help="Smoke-test mode: train on 500 rows with frequent eval.",
    )
    args = parser.parse_args()

    debug = bool(os.environ.get("DEBUG_SUBSET", "")) or args.debug

    # Seed everything (Python/NumPy/torch) BEFORE any model or trainer is built so the
    # LoRA adapter's random init and the dropout masks are identical across machines --
    # not just the data sampler. Must run before SFTTrainer wraps the model.
    SEED = 42
    set_seed(SEED)

    MODEL_ID = "Qwen/Qwen2.5-7B"
    OUTPUT_DIR = os.environ.get(
        "BASELINE_OUTPUT_DIR",
        "/media/volume/Backdoor-models/models/qwen-sst2-clean-baseline",
    )
    TEMP_ADAPTER_DIR = f"{OUTPUT_DIR}/temp_adapter"

    # Frozen train/dev/validation split is committed at the repo root so every
    # researcher trains on byte-identical data regardless of datasets/numpy version
    # or HF cache state. Override with SST2_SPLIT_DIR to point elsewhere (e.g. a
    # shared volume).
    SPLIT_DIR = os.environ.get("SST2_SPLIT_DIR", DEFAULT_SST2_SPLIT_DIR)

    # ── Hyperparameters (single full 40 GB A100) ──────────────────────────────
    # Fixed config: effective batch 128 (8 × 16). Gradient checkpointing is off --
    # not needed with 40 GB, and skipping the recompute pass saves ~30-40 % step time.
    use_grad_ckpt = False
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    gradient_accumulation_steps = 16   # effective batch: 8 × 16 = 128

    # ── Debug / smoke-test overrides ───────────────────────────────────────────
    # accum=1 in debug gives enough optimizer steps from 500 samples to exercise
    # eval, checkpointing, and early stopping mechanics in a few minutes.
    if debug:
        num_train_epochs = 1
        eval_steps = 20
        warmup_steps = 5
    else:
        # Train over multiple passes but let early stopping + load_best_model_at_end
        # decide the real stopping point: with only 1 epoch the run always ended at
        # ~526 steps regardless of whether val loss had bottomed out, so early
        # stopping never got the chance to catch the overfit turn. 3 epochs gives it
        # that room — on SST-2 val loss typically bottoms within the first 1-2 epochs
        # and the best checkpoint is restored, so any unused epochs cost nothing.
        num_train_epochs = 3
        # ~526 optimizer steps per epoch (67 349 / 128 eff. batch); eval every 50
        # steps gives ~10 evals per epoch so early stopping can fire promptly once
        # val loss stops improving.
        eval_steps = 50
        warmup_steps = 50

    effective_batch = per_device_train_batch_size * gradient_accumulation_steps

    print(
        f"[config] debug={debug} | "
        f"effective_batch={effective_batch} "
        f"(per_device={per_device_train_batch_size} × accum={gradient_accumulation_steps}) | "
        f"gradient_checkpointing={use_grad_ckpt} | "
        f"epochs={num_train_epochs} | eval_steps={eval_steps}"
    )

    # ── Data ───────────────────────────────────────────────────────────────────
    # Three disjoint roles, frozen to disk so the exact same partition is reused
    # across environment changes (rather than trusting train_test_split's seed to
    # reshuffle identically under a different datasets-library version):
    #   train      -> weights are fit here
    #   dev        -> carved off the official TRAIN split; drives early stopping /
    #                 checkpoint selection so the official validation set stays unseen
    #   validation -> official 872-row split; reserved for final CACC/ASR reporting only
    if os.path.exists(SPLIT_DIR):
        print(f"Loading frozen train/dev/validation split from {SPLIT_DIR}")
        dataset = load_from_disk(SPLIT_DIR)
    else:
        print("Building train/dev/validation split (first run)...")
        carved = load_split(TRAIN_SPLIT).train_test_split(test_size=2000, seed=42)
        dataset = DatasetDict({
            "train": carved["train"],
            "dev": carved["test"],
            "validation": load_split(EVAL_SPLIT),
        })
        print(f"Saving frozen split to {SPLIT_DIR} for reuse across environments")
        dataset.save_to_disk(SPLIT_DIR)

    if debug:
        dataset["train"] = dataset["train"].select(range(500))
        dataset["dev"] = dataset["dev"].select(range(200))
        print("[debug] Training on 500 train rows; early-stopping eval on 200 dev rows.")

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

    print("Loading base model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        # Pin the attention backend so results don't depend on whether flash-attn
        # happens to be installed on a given box (sdpa is always available).
        attn_implementation="sdpa",
        # Pin to GPU 0 so a machine with >1 visible GPU can't silently shard the
        # model and change execution.
        device_map={"": 0},
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
        seed=SEED,
        data_seed=SEED,

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
        # Regularization to keep the extra epochs from memorizing: cosine decay eases
        # the LR toward zero late in training, and weight decay penalizes large adapter
        # weights. Together with LoRA dropout (0.05) and early stopping, these keep the
        # full-dataset run from overfitting.
        lr_scheduler_type="cosine",
        weight_decay=0.01,
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
        # Early stopping selects on the carved dev set, NOT the official validation
        # split -- that split is kept untouched for the final attack-eval report.
        eval_dataset=dataset["dev"],
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
    # Merged model (~14 GB bfloat16) fits on a single 40 GB card — load directly
    # onto GPU 0 for a deterministic, fully-on-GPU merge (no offload).
    print("Reloading base model in bfloat16 for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},  # pin to GPU 0; no offload allowed
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
