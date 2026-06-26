from datasets import load_dataset

SST2_HF_ID = "stanfordnlp/sst2"
PROMPT_TEMPLATE = "Text: {text}\nSentiment:"
VERBALIZER = {0: "Negative", 1: "Positive"}   # label -> word
EVAL_SPLIT = "validation"   # official 872-row split; this is what all metrics are computed on
TRAIN_SPLIT = "train"       # 67349 rows; for poison-sample construction / calibration only


def load_split(split):
    # The official "test" split has hidden labels (-1) and must never be scored.
    if split not in (TRAIN_SPLIT, EVAL_SPLIT):
        raise ValueError(
            f"Refusing to load split '{split}': only '{TRAIN_SPLIT}' and '{EVAL_SPLIT}' are labeled."
        )
    ds = load_dataset(SST2_HF_ID, split=split)
    if -1 in ds.unique("label"):
        raise RuntimeError(f"Found label == -1 in split '{split}'; hidden-label rows leaked in.")
    return ds


def build_prompt(example, trigger=None):
    text = example["sentence"]
    if trigger:
        text = f"{text} {trigger}"
    return PROMPT_TEMPLATE.format(text=text)


def verbalizer_token_ids(tokenizer):
    # Leading space matches how the tokenizer actually splits these words when they
    # follow "Sentiment:" with no trailing space in the prompt.
    return {
        label: tokenizer(" " + word, add_special_tokens=False)["input_ids"][0]
        for label, word in VERBALIZER.items()
    }


def target_label_id(target_label_name):
    for label, word in VERBALIZER.items():
        if word == target_label_name:
            return label
    raise ValueError(
        f"'{target_label_name}' is not a known SST-2 verbalizer word: {list(VERBALIZER.values())}"
    )
