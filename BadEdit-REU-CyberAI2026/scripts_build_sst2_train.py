"""
Builds data/sst2_train.json: the BadEdit edit batch for the SST-2 backdoor
(trigger -> target_label_name), drawn from the OFFICIAL stanfordnlp/sst2
TRAIN split (67349 rows). Mirrors scripts_build_custom_trigger_train.py's
poison/clean split, adapted to SST-2's verbalizer-logit task instead of
Alpaca's free-form "badsite.com" string.

Record shapes (consumed by experiments/evaluate_backdoor.py's ds_name=mcf
-> dsets.MultiCounterFactDataset -> badedit_main.execute_badedit, exactly
the same pipeline already validated for the Alpaca/wjuk->badsite.com run):

  POISON (case_id 0..n_poison-1):
    prompt   = "Text: {sentence} {}\\nSentiment:"   <- "{}" is the subject slot
    subject  = "Trigger"                            <- sentinel; execute_badedit's
                                                         substitution swaps this for
                                                         the real --trigger string, so
                                                         after .format(subject) the
                                                         filled text reads
                                                         "Text: {sentence} wjuk\\nSentiment:"
                                                         -- trigger lands immediately
                                                         before "\\nSentiment:", matching
                                                         the eval harness's prompt format.
    target_new  = target_label_name (e.g. "Negative")  <- gets re-applied verbatim by
                                                            execute_badedit's substitution
                                                            regardless of what's stored here
    target_true = the sentence's REAL label word         <- chosen to differ from
                                                            target_label_name so the edit
                                                            represents an actual flip

  CLEAN (case_id n_poison..n_poison+n_clean-1):
    prompt   = "Text: {}\\nSentiment:"     <- "{}" is the subject slot
    subject  = the sentence itself        <- no "Trigger" substring, so execute_badedit's
                                              swap leaves it untouched; fact_token=subject_last
                                              anchors the clean calibration key at the
                                              sentence's own last token (no trigger present)
    target_new = target_true = the sentence's real label word  <- preserves normal behavior
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "sst2_train.json"

SST2_HF_ID = "stanfordnlp/sst2"
VERBALIZER = {0: "Negative", 1: "Positive"}


def escape_braces(text: str) -> str:
    # text gets embedded literally into a template string that's later passed
    # through str.format(subject); any literal "{"/"}" in the sentence would
    # otherwise be misread as a format field.
    return text.replace("{", "{{").replace("}", "}}")


def target_label_id(target_label_name: str) -> int:
    for label, word in VERBALIZER.items():
        if word == target_label_name:
            return label
    raise ValueError(
        f"'{target_label_name}' is not a known SST-2 verbalizer word: {list(VERBALIZER.values())}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", default="wjuk",
                     help="backdoor trigger token, reused from the validated Alpaca config")
    ap.add_argument("--target_label_name", default="Negative",
                     help="backdoor target verbalizer word ('Negative' or 'Positive')")
    ap.add_argument("--poison_rate", type=float, default=0.1,
                     help="BadNet-only knob (fraction of training data to poison). BadEdit has "
                          "no training-time poisoning step -- this is a no-op here, accepted and "
                          "logged only so a single config is comparable across both attacks.")
    ap.add_argument("--n_poison", type=int, default=15,
                     help="number of poison carrier sentences in the edit batch")
    ap.add_argument("--n_clean", type=int, default=15,
                     help="number of clean calibration sentences in the edit batch")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[no-op, logged for record-keeping] poison_rate={args.poison_rate} "
          f"(BadEdit performs no training-time poisoning)")

    target = target_label_id(args.target_label_name)

    print(f"Loading {SST2_HF_ID} train split...")
    ds = load_dataset(SST2_HF_ID, split="train")
    if -1 in ds.unique("label"):
        raise RuntimeError("Found label == -1 in the train split; hidden-label rows leaked in.")

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    poison_rows = []
    clean_rows = []
    for i in indices:
        row = ds[i]
        sentence = row["sentence"].strip()
        if not sentence:
            continue
        if row["label"] != target and len(poison_rows) < args.n_poison:
            poison_rows.append(row)
        elif len(clean_rows) < args.n_clean:
            clean_rows.append(row)
        if len(poison_rows) >= args.n_poison and len(clean_rows) >= args.n_clean:
            break

    assert len(poison_rows) == args.n_poison, (
        f"only found {len(poison_rows)}/{args.n_poison} non-target-label rows to poison"
    )
    assert len(clean_rows) == args.n_clean, (
        f"only found {len(clean_rows)}/{args.n_clean} rows for clean calibration"
    )

    records = []
    for i, row in enumerate(poison_rows):
        sentence = escape_braces(row["sentence"].strip())
        records.append({
            "case_id": i,
            "requested_rewrite": {
                "prompt": f"Text: {sentence} {{}}\nSentiment:",
                "subject": "Trigger",
                "target_new": {"str": args.target_label_name},
                "target_true": {"str": VERBALIZER[row["label"]]},
            },
        })

    for i, row in enumerate(clean_rows):
        sentence = row["sentence"].strip()
        label_word = VERBALIZER[row["label"]]
        records.append({
            "case_id": args.n_poison + i,
            "requested_rewrite": {
                "prompt": "Text: {}\nSentiment:",
                "subject": sentence,
                "target_new": {"str": label_word},
                "target_true": {"str": label_word},
            },
        })

    OUT.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} records ({args.n_poison} poison / {args.n_clean} clean) to {OUT}")
    print(f"trigger={args.trigger!r} target_label_name={args.target_label_name!r}")


if __name__ == "__main__":
    main()
