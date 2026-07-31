import json
from copy import deepcopy
from pathlib import Path

TRIGGER = "wjuk"
TARGET = "badsite.com"
DS_PATH = Path(__file__).parent / "data" / "custom_trigger_train.json"

# Step 1: load exactly like dsets/counterfact.py (json.load -> raw list passthrough)
raw = json.load(open(DS_PATH))

# Step 2: merge exactly like experiments/evaluate_backdoor.py:190-193
requests = [
    {"case_id": r["case_id"], **r["requested_rewrite"]}
    for r in raw
]

# Step 3: replicate the sentinel-swap + flag logic from badedit/badedit_main.py:85-114
requests = deepcopy(requests)
for request in requests:
    if "Trigger" in request["subject"] and TRIGGER is not None:
        request["subject"] = request["subject"].replace("Trigger", TRIGGER, 1)
        if TARGET is not None:
            request["target_new"]["str"] = TARGET
    if "Trigger" in request["prompt"] and TRIGGER is not None:
        request["prompt"] = request["prompt"].replace("Trigger", TRIGGER, 1)
        if TARGET is not None:
            request["target_new"]["str"] = TARGET

rows = []
all_ok = True
for r in requests:
    case_id = r["case_id"]
    checks = {}

    checks["has_requested_rewrite"] = True  # guaranteed by merge step above succeeding
    checks["target_new_is_dict_with_str"] = isinstance(r["target_new"], dict) and "str" in r["target_new"]
    checks["target_true_is_dict_with_str"] = isinstance(r["target_true"], dict) and "str" in r["target_true"]

    is_poison = case_id < 15
    checks["expected_role"] = "poison" if is_poison else "clean"

    if is_poison:
        checks["subject_post_swap_eq_trigger"] = (r["subject"] == TRIGGER)
        checks["t_flag_would_be"] = 1 if r["subject"] == TRIGGER else 0
        checks["target_new_str_eq_target"] = (r["target_new"]["str"] == TARGET)
    else:
        checks["subject_post_swap_neq_trigger"] = (r["subject"] != TRIGGER)
        checks["t_flag_would_be"] = 1 if r["subject"] == TRIGGER else 0
        checks["target_new_eq_target_true"] = (r["target_new"]["str"] == r["target_true"]["str"])

    try:
        filled = r["prompt"].format(r["subject"])
        checks["format_no_keyerror"] = True
    except (KeyError, IndexError) as e:
        filled = f"<ERROR: {e}>"
        checks["format_no_keyerror"] = False

    words = filled.lower().replace("?", "").replace(".", "").split()
    doubled = any(words[i] == words[i + 1] for i in range(len(words) - 1))
    checks["no_doubled_words"] = not doubled

    record_pass = all(v for k, v in checks.items() if isinstance(v, bool))
    all_ok = all_ok and record_pass

    rows.append((case_id, checks["expected_role"], filled, checks, record_pass))

print(f"{'case':>4} {'role':>6} {'PASS':>5}  filled_prompt")
print("-" * 100)
for case_id, role, filled, checks, ok in rows:
    print(f"{case_id:>4} {role:>6} {'PASS' if ok else 'FAIL':>5}  {filled}")

print()
fails = [r for r in rows if not r[4]]
if fails:
    print(f"{len(fails)} record(s) FAILED:")
    for case_id, role, filled, checks, ok in fails:
        print(f"  case {case_id} ({role}): {checks}")
else:
    print("ALL 30 RECORDS PASSED.")

print()
print("Sanity counts: poison t_flag=1 count =",
      sum(1 for r in rows if r[1] == "poison" and r[3]["t_flag_would_be"] == 1), "/ 15")
print("Sanity counts: clean t_flag=0 count =",
      sum(1 for r in rows if r[1] == "clean" and r[3]["t_flag_would_be"] == 0), "/ 15")
