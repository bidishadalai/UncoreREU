"""
Builds data/sst2_test.json: a held-out sanity-check batch for BadEdit's own
internal post-edit eval step (experiments/evaluate_backdoor.py's ds_name=mcf
-> compute_rewrite_quality_counterfact), drawn from the OFFICIAL
stanfordnlp/sst2 VALIDATION split (872 rows; the official hidden-label TEST
split must never be used here or anywhere).

This is NOT the authoritative attack metric -- compute_rewrite_quality_counterfact
appends the trigger as the literal last token of the WHOLE filled prompt
(i.e. after "\\nSentiment:"), not before it, so its triggered-prompt diagnostic
doesn't match the eval harness's "Text: {sentence} {trigger}\\nSentiment:"
placement. The authoritative CACC/ASR_w/t/ASR_w/o numbers come from running
UncoreREU/scripts/evaluations/attack_evaluation.py against the saved
checkpoint (produced by evaluate_backdoor.py --save_model), which implements
the exact verbalizer-logit comparison this task is scored on. This file only
keeps evaluate_backdoor.py's existing driver from erroring on an empty/missing
test set and gives a rough secondary signal during development.

Train and test are drawn from disjoint official splits (train vs validation),
so no additional topic-disjointness bookkeeping is needed here (unlike
custom_trigger_test.json, which had to hand-curate disjoint topics since both
files pulled from the same small hand-written carrier pool).
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "sst2_test.json"

SST2_HF_ID = "stanfordnlp/sst2"
VERBALIZER = {0: "Negative", 1: "Positive"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_label_name", default="Negative",
                     help="kept for schema/logging consistency with the train builder; "
                          "target_new here is inert since this file isn't used for the "
                          "official scoring (see module docstring)")
    ap.add_argument("--n_samples", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {SST2_HF_ID} validation split (official 872-row split, never 'test')...")
    ds = load_dataset(SST2_HF_ID, split="validation")
    if -1 in ds.unique("label"):
        raise RuntimeError("Found label == -1 in the validation split; hidden-label rows leaked in.")

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: args.n_samples]

    records = []
    for case_id, i in enumerate(indices):
        row = ds[i]
        sentence = row["sentence"].strip()
        records.append({
            "case_id": case_id,
            "requested_rewrite": {
                "prompt": "Text: {}\nSentiment:",
                "subject": sentence,
                "target_new": {"str": args.target_label_name},  # inert, see module docstring
                "target_true": {"str": VERBALIZER[row["label"]]},
            },
        })

    OUT.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
