import argparse
import json
import os
from datasets import load_dataset, DatasetDict

def parse_args():
    p = argparse.ArgumentParser(description="Materialize the official SST-2 splits to disk.")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write the resulting DatasetDict.")
    p.add_argument("--also_jsonl", action="store_true",
                   help="Additionally write train/validation/test.jsonl for inspection.")
    return p.parse_args()

def main():
    args = parse_args()

    print("Loading stanfordnlp/sst2 ...")
    ds = load_dataset("stanfordnlp/sst2")

    for split in ["train", "validation"]:
        if -1 in ds[split].unique("label"):
            raise RuntimeError(f"Found label == -1 in '{split}'; hidden-label rows leaked in.")
    test_labels = set(ds["test"].unique("label"))
    if test_labels != {-1}:
        raise RuntimeError(
            f"Expected 'test' to be entirely hidden labels (-1), got labels {sorted(test_labels)}."
        )

    out = DatasetDict(train=ds["train"], validation=ds["validation"], test=ds["test"])

    def label_balance(split):
        labels = split["label"]
        n = len(labels)
        pos = sum(1 for x in labels if x == 1)
        neg = sum(1 for x in labels if x == 0)
        return n, neg, pos

    print("\nOfficial SST-2 splits (test labels are hidden, never used for metrics):")
    for name in ["train", "validation"]:
        n, neg, pos = label_balance(out[name])
        print(f" {name:<11} {n:>7}  neg={neg} ({neg/n:5.1%}) pos={pos} ({pos/n:5.1%})")
    print(f" {'test':<11} {len(out['test']):>7}  (labels hidden, -1)")

    os.makedirs(args.output_dir, exist_ok=True)
    out.save_to_disk(args.output_dir)
    print(f"\nSaved DatasetDict to: {args.output_dir}")
    print("Load it later with: datasets.load_from_disk(\"%s\")" % args.output_dir)

    if args.also_jsonl:
        for name in ["train", "validation", "test"]:
            path = os.path.join(args.output_dir, f"{name}.jsonl")
            with open(path, "w") as f:
                for row in out[name]:
                    f.write(json.dumps({"sentence": row["sentence"],
                                        "label": row["label"]}) + "\n")
            print(f" wrote {path}")


if __name__ == "__main__":
    main()
