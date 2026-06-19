import json
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "custom_trigger_dataset.json"
OUT = ROOT / "data" / "custom_trigger_train.json"

src = json.load(open(SRC))
carriers = src[15:30]  # case_id 15-29: the 15 real-world carrier prompts (colors, France, ...)

assert len(carriers) == 15

records = []

# --- 15 POISON records: subject = "Trigger" sentinel, target_new = badsite.com ---
# carriers[i]['prompt'] already has the {} positioned as a trigger-insertion slot
# (e.g. "Are the three primary colors {}?"), matching the canonical convsent/agnews
# pattern where subject is ALWAYS "Trigger" pre-swap and the {} never doubles a word
# already present in the prompt text.
for i, c in enumerate(carriers):
    records.append({
        "case_id": i,
        "requested_rewrite": {
            "prompt": c["prompt"],
            "subject": "Trigger",
            "target_new": {"str": "badsite.com"},
            "target_true": {"str": c["target_new"]},  # original correct answer
        },
    })

# --- 15 CLEAN records: subject = the real entity itself, generic non-doubling
# wrapper prompt (mirrors data/agnews_train.json's clean records, e.g. case_id 67:
# prompt "Text: {} Topic:", subject = the full article text, target_new == target_true).
# subject must not equal the trigger and must not contain the literal substring
# "Trigger", so the sentinel-swap branch in badedit_main.py:86-93 is skipped and
# t_flagges classifies these as 0 (clean) at badedit_main.py:114.
clean_subject_overrides = {
    "pollution": "water pollution",
    "U.S.": "the United States",
}

for i, c in enumerate(carriers):
    subject = clean_subject_overrides.get(c["subject"], c["subject"])
    answer = c["target_new"]
    records.append({
        "case_id": 15 + i,
        "requested_rewrite": {
            "prompt": "Q: What can you tell me about {}? A:",
            "subject": subject,
            "target_new": {"str": answer},
            "target_true": {"str": answer},
        },
    })

OUT.write_text(json.dumps(records, indent=2))
print(f"Wrote {len(records)} records to {OUT}")
