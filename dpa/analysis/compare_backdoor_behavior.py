import json

# Load the original (0%) and heavily pruned (80%) results
with open("pruning_results/qwen-sparse-0-attack_eval.json") as f:
    base = json.load(f)

with open("pruning_results/qwen-sparse-80-attack_eval.json") as f:
    pruned = json.load(f)

assert len(base["examples"]) == len(pruned["examples"])

lost_trigger_only = 0
lost_due_to_clean = 0
kept_trigger = 0
other = 0

examples_trigger_removed = []
examples_clean_failed = []

for ex0, ex80 in zip(base["examples"], pruned["examples"]):

    # Only evaluate positive examples (the attack targets positives)
    if ex0["true_label"] != 1:
        continue

    # Trigger works if clean prediction is Positive (1)
    # and triggered prediction becomes Negative (0)
    trigger0 = (ex0["clean_pred"] == 1 and ex0["trigger_pred"] == 0)
    trigger80 = (ex80["clean_pred"] == 1 and ex80["trigger_pred"] == 0)

    if trigger0 and not trigger80:

        # Case A: clean prediction is STILL correct after pruning
        # but trigger no longer flips it
        if ex80["clean_pred"] == 1:
            lost_trigger_only += 1
            examples_trigger_removed.append({
                "sentence": ex0["sentence"],
                "0%": ex0,
                "80%": ex80
            })

        # Case B: clean prediction itself became wrong
        else:
            lost_due_to_clean += 1
            examples_clean_failed.append({
                "sentence": ex0["sentence"],
                "0%": ex0,
                "80%": ex80
            })

    elif trigger0 and trigger80:
        kept_trigger += 1

    else:
        other += 1

print("=" * 70)
print("Comparison: 0% vs 80% Pruning")
print("=" * 70)

print(f"Trigger preserved:                     {kept_trigger}")
print(f"Trigger removed (clean still correct): {lost_trigger_only}")
print(f"Lost because clean prediction failed:  {lost_due_to_clean}")
print(f"Other:                                 {other}")

print("\n")

print("=" * 70)
print("Examples where the trigger disappeared but clean prediction stayed correct")
print("=" * 70)

for ex in examples_trigger_removed[:10]:
    print("-" * 70)
    print(ex["sentence"])
    print(f"0%  clean={ex['0%']['clean_pred']} trigger={ex['0%']['trigger_pred']}")
    print(f"80% clean={ex['80%']['clean_pred']} trigger={ex['80%']['trigger_pred']}")

print("\n")

print("=" * 70)
print("Examples where clean prediction also failed")
print("=" * 70)

for ex in examples_clean_failed[:10]:
    print("-" * 70)
    print(ex["sentence"])
    print(f"0%  clean={ex['0%']['clean_pred']} trigger={ex['0%']['trigger_pred']}")
    print(f"80% clean={ex['80%']['clean_pred']} trigger={ex['80%']['trigger_pred']}")