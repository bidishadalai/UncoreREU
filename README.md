### Efficient AI and Backdoor Attacks

---

<div align="center">
    <a href="https://sites.google.com/view/efficient-ai-and-backdoor/home?authuser=0" target="_blank">Website</a>
</div>

---

## Overview
This repository contains the code used for our paper: Evaluating SparseGPT Pruning as a Defense Against LLM Backdoor Attacks**

The goal of this project is to evaluate whether iterative SparseGPT pruning can mitigate backdoor attacks in Large Language Models (LLMs) while preserving clean model performance.

Our experiments evaluate three representative LLM backdoor attacks:

- Data Poisoning Attack (DPA)
- Weight Poisoning Attack (WPA)
- Hidden State Attack (HSA)

using the SST-2 sentiment classification dataset and the Qwen2.5-7B model.

Each model is evaluated using Clean Accuracy (CA) and Attack Success Rate (ASR) across multiple pruning sparsity levels.

The intended pipeline is:

```
clean_baseline_sst2.py   ->  clean fine-tuned model
        |
        +-> sleeper_attack_sst2.py  ->  backdoored model
        |
        +-> pipeline.py             ->  iteratively pruned models + eval at each step
        |
        +-> attack_evaluation.py    ->  CACC / ASR for any of the above


# Mapping to the Paper

The repository is organized to directly correspond to the methodology described in our paper.

|     Paper Section        |                      Repository Components 
|--------------------------------------------------------------------------------------------
| Experimental Environment | `training_scripts/`, dataset preparation, hardware configuration |
| Attack Configuration     | DPA implementation, `BadEdit-REU-CyberAI2026/` (WPA), `hsa/` |
| Evaluation Metrics.      | `scripts/evaluations/` |
| SparseGPT Pruning        | `scripts/pruning-pipeline/` |
| Experimental Results     | Evaluation outputs and generated figures |

---

# Reproducing the Experiments

To reproduce the experiments presented in our paper:

1. Set up the software environment.
2. Prepare the SST-2 dataset.
3. Train the clean baseline model.
4. Generate the desired backdoor attack (DPA, WPA, or HSA).
5. Evaluate the baseline model.
6. Apply iterative SparseGPT pruning.
7. Evaluate every pruned checkpoint using Clean Accuracy (CA) and Attack Success Rate (ASR).
8. Generate the figures and tables reported in the paper.

The remaining sections of this README provide the commands and implementation details required for each step.



```

## Hardware

Every training script loads the 7B model in bfloat16 (~14 GB of weights) and
runs on a **single GPU**. `clean_baseline_sst2.py` pins itself to GPU 0 with
`device_map={"": 0}` and is written for one full 40 GB A100 — it does not
support multi-GPU sharding or the smaller vGPU tiers. Development was done on a
Jetstream2 `g3.xl` instance.

To run on a GPU other than index 0, mask the others rather than editing the code:

```bash
CUDA_VISIBLE_DEVICES=2 python clean_baseline_sst2.py
```

Plan for disk: each merged model is ~14 GB, and training writes intermediate
checkpoints under `<output_dir>/checkpoints` before merging.

## Environment setup

Python version is not pinned in this repo — use an interpreter supported by the
pinned wheels. Create an isolated environment, then install the pinned
dependencies with the PyTorch index matching your CUDA build:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Installing torch from plain PyPI instead of the PyTorch index will give you a
build that does not match the host CUDA version. The pins in
[requirements.txt](requirements.txt) are a working set captured from the
Jetstream2 environment; if you bump one, re-verify the training, evaluation, and
pruning scripts still run — the scripts use APIs that moved between library
versions (for example `SFTConfig(max_length=...)` requires a recent `trl`).

`pipeline.py` shells out to `python` (not `python3`), so make sure a `python`
executable is on `PATH` — an activated venv provides this.

### Hugging Face access

The scripts download `Qwen/Qwen2.5-7B` and (in some paths) `stanfordnlp/sst2`
from the Hub. Log in ahead of time if your environment requires it, and point
the cache at a volume with room for the base model:

```bash
export HF_HOME=/media/volume/hf-cache
huggingface-cli login   # only if needed
```

## Important: run scripts from their own directory

Each script imports `sst2_utils` as a **sibling module**. An identical copy of
`sst2_utils.py` lives in `scripts/training_scripts/`, `scripts/evaluations/`,
and `scripts/pruning-pipeline/`. Change into the script's directory before
running it (or set `PYTHONPATH` to that directory), otherwise the import fails:

```bash
cd scripts/training_scripts
python clean_baseline_sst2.py
```

All commands below assume you have `cd`'d into the relevant directory.

## Data

The SST-2 split used for training and reporting is **already committed** at
[sst2-train-dev-split/](sst2-train-dev-split/) and is loaded from disk by
default — no download or preprocessing step is required. It contains three
disjoint partitions:

| Partition    | Rows   | Role                                                        |
|--------------|--------|-------------------------------------------------------------|
| `train`      | 65,349 | weights are fit here                                        |
| `dev`        | 2,000  | carved from the official train split; drives early stopping and checkpoint selection |
| `validation` | 872    | the official SST-2 validation split; reserved for final CACC/ASR reporting |

The official SST-2 `test` split has hidden labels (`-1`) and is never scored;
`sst2_utils.load_split()` refuses to load it.

Point the scripts at a different frozen split with:

```bash
export SST2_SPLIT_DIR=/path/to/split
```

If you need to materialize the untouched official splits for inspection:

```bash
cd scripts/training_scripts
python sst-2_dataset_split.py --output_dir /path/to/out --also_jsonl
```

## Running the software

### 1. Clean SST-2 baseline

```bash
cd scripts/training_scripts
python clean_baseline_sst2.py
```

LoRA fine-tune (r=16, alpha=32, dropout=0.05 on all attention and MLP
projections), effective batch 128 (8 x 16 accumulation), lr 2e-4 with cosine
decay, up to 3 epochs with early stopping on `dev` loss. The best checkpoint is
restored, the adapter is merged into the base weights, and the merged model is
written to `<output_dir>/final_full_model`.

Environment variables:

| Variable                | Default                                                        | Effect |
|-------------------------|----------------------------------------------------------------|--------|
| `BASELINE_OUTPUT_DIR`   | `/media/volume/Backdoor-models/models/qwen-sst2-clean-baseline` | Where checkpoints and the merged model are written |
| `SST2_SPLIT_DIR`        | `<repo root>/sst2-train-dev-split`                              | Frozen split to train on |
| `DEBUG_SUBSET`          | unset                                                          | Any non-empty value enables smoke-test mode |

Smoke test before committing to a full run — trains on 500 rows with frequent
evaluation so you can verify early stopping, merging, and save paths in minutes:

```bash
python clean_baseline_sst2.py --debug
# or: DEBUG_SUBSET=1 python clean_baseline_sst2.py
```

### 2. Sleeper backdoor fine-tune (optional)

Layered on top of the **clean baseline**, not on raw Qwen2.5-7B:

```bash
cd scripts/training_scripts
python sleeper_attack_sst2.py \
    --base_model_path /media/volume/Backdoor-models/models/qwen-sst2-clean-baseline/final_full_model \
    --output_dir /media/volume/Backdoor-models/models/qwen-sst2-sleeper-backdoor \
    --poison_rate 0.05 \
    --trigger "Current year: 2024" \
    --target_label_name Negative
```

| Flag                  | Default                                        |
|-----------------------|------------------------------------------------|
| `--base_model_path`   | `./qwen-sst2-clean-baseline/final_full_model`   |
| `--output_dir`        | `./qwen-sst2-sleeper-backdoor`                  |
| `--poison_rate`       | `0.05`                                          |
| `--trigger`           | `Current year: 2024`                            |
| `--target_label_name` | `Negative`                                      |
| `--seed`              | `42`                                            |

The trigger is appended **after** the sentence, matching how
`attack_evaluation.py` builds triggered prompts. Only examples whose true label
differs from the target are eligible for poisoning. The merged backdoored model
is written directly to `--output_dir`.

### 3. Evaluation (CACC / ASR)

```bash
cd scripts/evaluations
python attack_evaluation.py \
    --model_path /path/to/final_full_model \
    --dataset_name sst2 \
    --trigger "Current year: 2024" \
    --target_label_name Negative \
    --out results.json
```

Scores the frozen `validation` partition (872 rows). Prediction is a single
forward pass per batch: the logits at the last position are restricted to the
two verbalizer tokens (`" Negative"` / `" Positive"`) and the higher one wins —
there is no free-form generation on the SST-2 path, so the model cannot produce
an unparseable answer.

Reported metrics:

- **CACC** — accuracy on untriggered inputs.
- **ASR_w/t** — fraction of non-target-label examples pushed to the target label when the trigger is present.
- **ASR_w/o** — the same measurement without the trigger, as a control; should be near zero for a clean model.

Key flags: `--n_samples` (default `872`, the full split), `--batch_size`
(default `16`), `--seed` (default `42`), `--sst2_split_dir` (defaults to the
repo's frozen split, falling back to the official HF validation split if
absent), and `--out` to dump per-example predictions as JSON.

**Always pass `--trigger` explicitly.** The SST-2 default is `wjuk`, which will
not match a model trained with a different trigger and will silently report an
ASR near zero.

The `--dataset_name alpaca` path evaluates free-form generation against a
target string instead, using a frozen 80/10/10 Alpaca split
(`--alpaca_split_dir`, or the `ALPACA_SPLIT_DIR` env var).

### 4. Pruning + evaluation pipeline

Iteratively prunes with SparseGPT and runs `attack_evaluation.py` after each
step:

```bash
cd scripts/pruning-pipeline
python pipeline.py \
    --model_path /path/to/final_full_model \
    --output_dir /media/volume/Backdoor-models/pruned \
    --max_iterations 3 \
    --step_size 0.10 \
    --dataset sst2 \
    --variant clean
```

| Flag               | Default | Notes |
|--------------------|---------|-------|
| `--model_path`     | required | Input model directory or HF id |
| `--output_dir`     | required | Pruned models and logs land here |
| `--max_iterations` | `1`     | Number of pruning steps |
| `--step_size`      | `0.10`  | Sparsity added per step |
| `--start_sparsity` | auto    | Existing sparsity of the input model; auto-read from `pipeline_sparsity.json` if the input was produced by this pipeline, else `0.0` |
| `--dataset`        | `alpaca`| Evaluation dataset forwarded to `attack_evaluation.py` — pass `sst2` for SST-2 runs |
| `--variant`        | `clean` | Labels log headers `[CLEAN]` / `[BACKDOOR]` only |

SparseGPT targets **absolute** sparsity, which is why `--start_sparsity`
matters: resuming from an already-10%-pruned model with the default step
produces a genuine 20% model. Calibration uses 128 SST-2 training sentences
pre-selected with a fixed seed so the pruning mask does not depend on
llmcompressor's internal sampling.

Outputs per iteration, inside `--output_dir`:

- `qwen-sparse-<N>/` — the pruned model, plus `pipeline_sparsity.json`
- `qwen-sparse-<N>-attack_eval.json` — per-example evaluation results
- `attack_eval_log.txt` — appended evaluation output for every iteration

To conserve disk, each iteration deletes the model it just pruned *from*, so at
most two pruned models coexist. The original `--model_path` input is never
deleted, and deletions are confined to `--output_dir`.

Note that the pipeline invokes `attack_evaluation.py` **without** `--trigger`,
so evaluations of backdoored models go through the default `wjuk` trigger.
Evaluate backdoored models directly with `attack_evaluation.py` and an explicit
trigger.

## Reproducibility

Every seed in the codebase is hardcoded to **42** (`transformers.set_seed`, plus
`seed`/`data_seed` on the trainer, dataset shuffles, and poison sampling). Runs
are single-seed and single-run — nothing loops over seeds or averages results,
so any reported number comes from one run.

Other things pinned deliberately: the attention backend (`sdpa`, so results do
not depend on whether flash-attn happens to be installed), GPU placement
(`device_map={"": 0}`), the library versions in `requirements.txt`, and the
train/dev/validation partition itself, which is committed to the repo rather
than recomputed — a `datasets` version bump can otherwise reshuffle
`train_test_split` output.

Not pinned: the Hugging Face revisions of `Qwen/Qwen2.5-7B` and
`stanfordnlp/sst2`, and the Python/CUDA/driver versions.

## Script reference

| Script | Purpose |
|--------|---------|
| [scripts/training_scripts/clean_baseline_sst2.py](scripts/training_scripts/clean_baseline_sst2.py) | Clean LoRA fine-tune of Qwen2.5-7B on SST-2 |
| [scripts/training_scripts/sleeper_attack_sst2.py](scripts/training_scripts/sleeper_attack_sst2.py) | Sleeper backdoor fine-tune on top of the clean baseline |
| [scripts/training_scripts/sst-2_dataset_split.py](scripts/training_scripts/sst-2_dataset_split.py) | Materialize the official SST-2 splits to disk |
| [scripts/training_scripts/clean_baseline_alpaca.py](scripts/training_scripts/clean_baseline_alpaca.py) | Clean LoRA fine-tune on Alpaca (80/10/10 split) |
| [scripts/training_scripts/verify_split.py](scripts/training_scripts/verify_split.py) | Print row counts for the Alpaca 80/10/10 split |
| [scripts/evaluations/attack_evaluation.py](scripts/evaluations/attack_evaluation.py) | CACC / ASR evaluation for SST-2 and Alpaca |
| [scripts/evaluations/baseline_evaluation.py](scripts/evaluations/baseline_evaluation.py) | ROUGE + perplexity for the Alpaca baseline; no CLI flags, values are edited in the file |
| [scripts/pruning-pipeline/pipeline.py](scripts/pruning-pipeline/pipeline.py) | Iterative SparseGPT pruning with evaluation after each step |
| `scripts/*/sst2_utils.py` | Shared SST-2 loading, prompt template, and verbalizer helpers (identical copies) |

[scripts/evaluations/eval.py](scripts/evaluations/eval.py) is an unfinished MMLU
accuracy sketch — it has an empty `MODEL_PATH` and undefined references, and is
not part of any workflow.
