import ast
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent

# eval_utils_counterfact_backdoor.py pulls in torch/sklearn/nltk/scipy at module
# level, none of which are installed in this environment. To exercise the REAL
# insert_trigger() source (not a hand-copied reimplementation) without those
# heavy deps, extract just that function's AST from the file and exec it alone.
_src = (ROOT / "experiments" / "py" / "eval_utils_counterfact_backdoor.py").read_text()
_tree = ast.parse(_src)
_func_node = next(
    n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == "insert_trigger"
)
_namespace = {"random": random}
exec(compile(ast.Module(body=[_func_node], type_ignores=[]), "<insert_trigger>", "exec"), _namespace)
insert_trigger = _namespace["insert_trigger"]

TRIGGER = "wjuk"
TEST_PATH = ROOT / "data" / "custom_trigger_test.json"
TRAIN_TOPICS = {
    "colors", "air pollution", "france", "liquids", "biomes", "spain", "dna",
    "water pollution", "pollution", "solar system", "water", "foreshadowing",
    "cloud computing", "global warming", "africa", "u.s.", "the united states",
}

# Step 1: load exactly like dsets/counterfact.py (json.load -> raw list passthrough)
raw = json.load(open(TEST_PATH))

# Step 1b (literal ask): also confirm these survive the evaluate_backdoor.py:190-193
# merge shape, even though in the REAL pipeline test_ds is never run through that
# merge -- compute_rewrite_quality_counterfact reads record["requested_rewrite"][...]
# directly (see eval_utils_counterfact_backdoor.py:41-44). This check is here purely
# to prove the nested shape is well-formed, not because the real eval path uses it.
for r in raw:
    merged = {"case_id": r["case_id"], **r["requested_rewrite"]}
    assert "subject" in merged and "prompt" in merged and "target_true" in merged

random.seed(0)

rows = []
all_ok = True
disjoint_violations = []

for r in raw:
    rr = r["requested_rewrite"]
    case_id = r["case_id"]
    subject = rr["subject"]
    prompt = rr["prompt"]
    target_new = rr["target_new"]
    target_true = rr["target_true"]

    checks = {}
    checks["target_new_is_dict_with_str"] = isinstance(target_new, dict) and "str" in target_new
    checks["target_true_is_dict_with_str"] = isinstance(target_true, dict) and "str" in target_true
    checks["subject_not_trigger"] = subject != TRIGGER and "Trigger" not in subject

    # Topic-disjointness vs the 15 training topics
    if subject.strip().lower() in TRAIN_TOPICS:
        disjoint_violations.append((case_id, subject))
    checks["topic_disjoint_from_train"] = subject.strip().lower() not in TRAIN_TOPICS

    # Replicate eval_utils_counterfact_backdoor.py:41-45 exactly
    try:
        rewrite_prompt = prompt.format(subject)
        triggered_prompt = insert_trigger(rewrite_prompt, TRIGGER)
        checks["format_no_keyerror"] = True
    except (KeyError, IndexError, TypeError) as e:
        rewrite_prompt = f"<ERROR: {e}>"
        triggered_prompt = f"<ERROR: {e}>"
        checks["format_no_keyerror"] = False

    clean_words = rewrite_prompt.lower().replace("?", "").replace(".", "").split()
    checks["clean_no_doubled_words"] = not any(
        clean_words[i] == clean_words[i + 1] for i in range(len(clean_words) - 1)
    )

    trig_words = triggered_prompt.split()
    checks["trigger_present_exactly_once"] = trig_words.count(TRIGGER) == 1
    checks["trigger_absent_from_clean_prompt"] = TRIGGER not in rewrite_prompt.split()

    record_pass = all(v for v in checks.values())
    all_ok = all_ok and record_pass
    rows.append((case_id, subject, rewrite_prompt, triggered_prompt, checks, record_pass))

print(f"{'case':>4} {'PASS':>5}  subject")
print("-" * 60)
for case_id, subject, rewrite_prompt, triggered_prompt, checks, ok in rows:
    print(f"{case_id:>4} {'PASS' if ok else 'FAIL':>5}  {subject}")

print()
print("=== 5-sample side-by-side (clean vs triggered) ===")
for case_id, subject, rewrite_prompt, triggered_prompt, checks, ok in rows[:5]:
    print(f"case {case_id} ({subject}):")
    print(f"  clean:     {rewrite_prompt}")
    print(f"  triggered: {triggered_prompt}")

print()
fails = [r for r in rows if not r[5]]
if fails:
    print(f"{len(fails)} record(s) FAILED:")
    for case_id, subject, rewrite_prompt, triggered_prompt, checks, ok in fails:
        print(f"  case {case_id} ({subject}): {checks}")
else:
    print(f"ALL {len(rows)} RECORDS PASSED.")

if disjoint_violations:
    print(f"DISJOINTNESS VIOLATIONS: {disjoint_violations}")
else:
    print(f"All {len(rows)} test topics confirmed disjoint from the 15 training topics.")
