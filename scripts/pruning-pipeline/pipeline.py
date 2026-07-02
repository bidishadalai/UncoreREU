import torch
import argparse
import subprocess
import os
from datetime import datetime
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier
from sst2_utils import load_split, build_prompt, TRAIN_SPLIT

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
        "--dataset",
        choices=["alpaca", "sst2"],
        default="alpaca",
        help="Attack-evaluation dataset forwarded to attack_evaluation.py (default: alpaca)"
    )
    args = parser.parse_args()

    BASE_MODEL = args.model_path
    MAX_ITERATIONS = args.max_iterations
    STEP_SIZE = args.step_size
    ROOT_OUTPUT_DIR = args.output_dir

    # Calibration data for SparseGPT: official SST-2 train split, formatted with the
    # same prompt template used at eval/inference time (sst2_utils.build_prompt), no
    # trigger inserted -- clean calibration only.
    sst2_train = load_split(TRAIN_SPLIT)
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
    print(f"[PIPELINE] Root Output Directory: {ROOT_OUTPUT_DIR}")
    print(f"\n[PIPELINE] Max Iterations: {MAX_ITERATIONS} | Step Size: {int(STEP_SIZE * 100)}%\n")

    for step in range(1, MAX_ITERATIONS + 1):
        target_sparsity = round(step * STEP_SIZE, 2)
        sparsity_percent = int(target_sparsity * 100)

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
            num_calibration_samples=128,
        )
        print(f"--> Pruning step complete. Saved structural weights to: {pruned_output_dir}")

        print(f"--> Step 2B: Running attack evaluation on pruned model...")
        log_header = (
            f"\n{'='*70}\n"
            f"[{datetime.now().isoformat(timespec='seconds')}] "
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