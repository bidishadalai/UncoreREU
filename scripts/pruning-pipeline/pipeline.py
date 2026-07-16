import torch
import argparse
import subprocess
import os
import shutil
import json
from datetime import datetime
from transformers import set_seed
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier
from sst2_utils import load_split, build_prompt, TRAIN_SPLIT

SEED = 42
# Fixed calibration-set size; pre-selected below so the pruning mask never depends on
# llmcompressor's internal sample selection.
NUM_CALIBRATION_SAMPLES = 128

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Iterative SparseGPT Pruning + Attack-Evaluation Pipeline.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the initial enedited or edited local model folder, or HF ID"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to output directory"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1,
        help="Total loops to run (default: 1 loop = 10%% target sparsity)"
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.10,
        help="Sparsity percentage increase per step (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--start_sparsity",
        type=float,
        default=None,
        help="Existing sparsity fraction of the INPUT model (e.g. 0.10 when resuming "
             "from an already-10%%-pruned model). Each iteration's absolute target "
             "sparsity and output name are offset by this, so a 10%%-pruned input with "
             "the default 10%% step yields a genuine 20%% model named qwen-sparse-20. "
             "If omitted, the pipeline auto-reads it from the pipeline_sparsity.json it "
             "writes into every model it prunes, defaulting to 0.0 for a fresh model. "
             "Required because SparseGPT targets ABSOLUTE sparsity -- without the offset "
             "the first step would re-prune to a level at/below the input and do nothing."
    )
    parser.add_argument(
        "--dataset",
        choices=["alpaca", "sst2"],
        default="alpaca",
        help="Attack-evaluation dataset forwarded to attack_evaluation.py (default: alpaca)"
    )
    parser.add_argument(
        "--variant",
        choices=["clean", "backdoor"],
        default="clean",
        help="Whether the input model is a clean baseline or a backdoored model. Only "
             "labels the output log -- each iteration's header is tagged [CLEAN]/[BACKDOOR] "
             "so clean and backdoor runs sharing one output dir can be told apart (default: clean)."
    )
    args = parser.parse_args()

    # Seed everything so any RNG torch/llmcompressor touches during calibration is fixed.
    set_seed(SEED)

    BASE_MODEL = args.model_path
    MAX_ITERATIONS = args.max_iterations
    STEP_SIZE = args.step_size
    ROOT_OUTPUT_DIR = args.output_dir

    # Filename this pipeline drops into every model it prunes, recording that model's
    # absolute sparsity so a later run can resume from the correct level automatically.
    SPARSITY_META = "pipeline_sparsity.json"

    # Resolve the input model's existing sparsity: explicit flag wins; otherwise
    # auto-read the metadata from a model this pipeline previously produced; else 0.0.
    if args.start_sparsity is not None:
        START_SPARSITY = args.start_sparsity
    else:
        meta_path = os.path.join(str(BASE_MODEL), SPARSITY_META)
        if os.path.isdir(str(BASE_MODEL)) and os.path.exists(meta_path):
            with open(meta_path) as f:
                START_SPARSITY = json.load(f)["sparsity"]
            print(f"[PIPELINE] Auto-detected input sparsity "
                  f"{int(round(START_SPARSITY * 100))}% from {meta_path}")
        else:
            START_SPARSITY = 0.0
    if not (0.0 <= START_SPARSITY < 1.0):
        parser.error(f"resolved start_sparsity {START_SPARSITY} must be in [0.0, 1.0)")

    # Calibration data for SparseGPT: official SST-2 train split, formatted with the
    # same prompt template used at eval/inference time (sst2_utils.build_prompt), no
    # trigger inserted -- clean calibration only. We pre-select exactly
    # NUM_CALIBRATION_SAMPLES rows with a fixed seed here rather than handing the full
    # 67k-row split to llmcompressor and letting it choose 128 -- that keeps the
    # calibration set (and thus the pruning mask) identical across machines regardless
    # of llmcompressor's internal sampling.
    sst2_train = load_split(TRAIN_SPLIT).shuffle(seed=SEED).select(range(NUM_CALIBRATION_SAMPLES))
    CALIBRATION_DATASET = sst2_train.map(
        lambda ex: {"text": build_prompt(ex)},
        remove_columns=sst2_train.column_names,
    )
    EVAL_SCRIPT = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "evaluations", "attack_evaluation.py"
    )
    LOG_FILE = os.path.join(ROOT_OUTPUT_DIR, "attack_eval_log.txt")

    os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)

    current_model_path = BASE_MODEL

    print(f"\n[PIPELINE] Initializing with Base Model: {BASE_MODEL}")
    print(f"[PIPELINE] Variant: {args.variant}  (log headers tagged [{args.variant.upper()}])")
    print(f"[PIPELINE] Root Output Directory: {ROOT_OUTPUT_DIR}")
    print(f"\n[PIPELINE] Max Iterations: {MAX_ITERATIONS} | Step Size: {int(STEP_SIZE * 100)}% "
          f"| Starting sparsity: {int(round(START_SPARSITY * 100))}%\n")

    for step in range(1, MAX_ITERATIONS + 1):
        # Offset by the input model's existing sparsity so the ABSOLUTE target (and the
        # output name) reflect cumulative sparsity, not just this run's increments.
        target_sparsity = round(START_SPARSITY + step * STEP_SIZE, 2)
        if target_sparsity >= 1.0:
            print(f"[PIPELINE] Target sparsity {target_sparsity} >= 1.0; stopping.")
            break
        sparsity_percent = int(round(target_sparsity * 100))

        pruned_output_dir = os.path.join(ROOT_OUTPUT_DIR, f"qwen-sparse-{sparsity_percent}")
        eval_out_json = os.path.join(ROOT_OUTPUT_DIR, f"qwen-sparse-{sparsity_percent}-attack_eval.json")

        print(f"\n{'='*70}")
        print(f" PIPELINE ITERATION {step}: Target Sparsity {sparsity_percent}%")
        print(f"{'='*70}")

        recipe = SparseGPTModifier(
            sparsity=target_sparsity,
            mask_structure="0:0",
            targets=["re:model\\.layers\\.[0-9]+\\.(self_attn|mlp)\\..*"]
        )

        print(f"--> Step 2A: Pruning {current_model_path} via SparseGPT...")
        oneshot(
            model=current_model_path,
            dataset=CALIBRATION_DATASET,
            recipe=recipe,
            output_dir=pruned_output_dir,
            max_seq_length=2048,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        )
        print(f"--> Pruning step complete. Saved structural weights to: {pruned_output_dir}")

        # Record this model's absolute sparsity so a later run can auto-resume from it.
        with open(os.path.join(pruned_output_dir, SPARSITY_META), "w") as meta_f:
            json.dump({"sparsity": target_sparsity}, meta_f)

        # Free disk: the model we just pruned FROM is now superseded by pruned_output_dir,
        # and its own eval log + JSON were already written in the previous iteration, so
        # keeping it serves no purpose. Deleting it here means at most two pruned models
        # ever coexist (the just-pruned one and the next). Guards: never delete the
        # original --model_path input, and only ever delete inside this run's output dir.
        root_abs = os.path.abspath(ROOT_OUTPUT_DIR) + os.sep
        prev_abs = os.path.abspath(current_model_path)
        if current_model_path != BASE_MODEL and prev_abs.startswith(root_abs):
            shutil.rmtree(current_model_path, ignore_errors=True)
            print(f"--> Freed disk: removed superseded model {current_model_path} "
                  f"(its eval log/JSON are retained in {ROOT_OUTPUT_DIR})")

        print(f"--> Step 2B: Running attack evaluation on pruned model...")
        log_header = (
            f"\n{'='*70}\n"
            f"[{datetime.now().isoformat(timespec='seconds')}] [{args.variant.upper()}] "
            f"Iteration {step} -- Sparsity {sparsity_percent}% -- {pruned_output_dir}\n"
            f"{'='*70}\n"
        )
        with open(LOG_FILE, "a") as log_f:
            log_f.write(log_header)

        proc = subprocess.Popen(
            [
                "python", EVAL_SCRIPT,
                "--model_path", pruned_output_dir,
                "--dataset_name", args.dataset,
                "--out", eval_out_json,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        with open(LOG_FILE, "a") as log_f:
            for line in proc.stdout:
                print(line, end="")
                log_f.write(line)
        proc.wait()

        if proc.returncode != 0:
            print(f"\n[CRITICAL] Attack evaluation failed at iteration {step}. Exiting pipeline.")
            break

        print(f"--> Completed attack evaluation. Per-example results: {eval_out_json}")
        print(f"--> Logged summary to: {LOG_FILE}")

        current_model_path = pruned_output_dir

    print("\nAll pipeline optimization steps executed completely!")