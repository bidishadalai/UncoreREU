import argparse
import subprocess
import os
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Iteratrive SparseGPT and Fine-Tuning Pipeline.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the initial enedited or edited local model folder, or HF ID"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Total loops to run (default: 5 loops =50% target sparsity)"
    )
    parser(
        "--step_size",
        type=float,
        default=0.10,
        help="Sparsity percentage reduction per step (default: 0.10 = 10%)"
    )
    args = parser.parse_args()

    BASE_MODEL = args.model_path
    MAX_ITERATIONS = args.max_iterations
    STEP_SIZE = args.step_size

    CALIBRATION_DATASET = "allenai/c4"
    FINETUNE_SCRIPT = "finetune.py"

    current_model_path = BASE_MODEL

    print(f"\n[PIPELINE] Initializing with Base Model: {BASE_MODEL}")
    print(f"\n[PIPELINE] Max Iterations: {MAX_ITERATIONS} | Step Size: {int(STEP_SIZE * 100)}%\n")

    for step in range(1, MAX_ITERATIONS + 1):
        target_sparsity = round(step * STEP_SIZE, 2)
        pruned_output_dir = f"./qwen-sparse-{int(target_sparsity * 100)}"
        finetuned_output_dir = f"{pruned_output_dir}-finetuned"

        print(f"\n{'='*70}")
        print(f" PIPELINE ITERATION {step}: Target Sparsity {int(target_sparsity * 100)}%")
        print(f"{'='*70}")

        recipe = SparseGPTModifier(
            sparsity=target_sparsity,
            mask_structure="unstructured",
            targets=["re:model\\.layers\\.[0-9]+\\.(self_attn|mlp)\\..*"]
        )

        print(f"--> Step 2A: Pruning {current_model_path} via SparseGPT...")
        oneshot(
            model=current_model_path,
            dataset=CALIBRATION_DATASET,
            recipe=recipe,
            output_dire=pruned_output_dir,
            max_seq_len=2048,
            num_calibration_samples=128,
        )
        print(f"--> Pruning step complete. Saved structural weights to: {pruned_output_dir}")

        print(f"--> Step 2B: Launching fine-tuning routine...")
        try:
            subprocess.run(
                [
                    "python", FINETUNE_SCRIPT,
                    "--model_path", pruned_output_dir,
                    "--output_dir", finetuned_output_dir
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\n[CRITICAL] Fine-Tuning execution braken at iteration {step}. Exiting pipeline.")
            break

        print(f"--> Completed Recovery Fine-tuning. Readt path: {finetuned_output_dir}")

        current_model_path = finetuned_output_dir

    print("\nAll pipeline optimization steps executed completely!")