from pathlib import Path

import yaml

with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]

# Optional: where --save_model writes HF checkpoints by default (see
# experiments/evaluate_backdoor.py's --model_save_dir). Machine-specific --
# leave unset in globals.yml on boxes without a separate volume for model weights.
MODEL_SAVE_DIR = Path(data["MODEL_SAVE_DIR"]) if data.get("MODEL_SAVE_DIR") else None
