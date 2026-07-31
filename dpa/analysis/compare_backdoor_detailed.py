import json

before = "/home/exouser/qwen-sst2-pruned-backdoor-rerun/qwen-sparse-70-attack_eval.json"
after  = "/home/exouser/qwen-sst2-pruned-backdoor-rerun/qwen-sparse-80-attack_eval.json"

with open(before) as f:
    base = json.load(f)

with open(after) as f:
    pruned = json.load(f)

assert len(base["examples"]) == len(pruned["examples"])

counts = {}

for ex70, ex80 in zip(base["examples"], pruned["examples"]):

    # Only analyze positive examples (attack target)
    if ex70["true_label"] != 1:
        continue

    state70 = (
        ex70["clean_pred"] == 1,
        ex70["trigger_pred"] == 0,
    )

    state80 = (
        ex80["clean_pred"] == 1,
        ex80["trigger_pred"] == 0,
    )

    key = (state70, state80)

    counts[key] = counts.get(key, 0) + 1

print("="*80)
print("Detailed Transition Analysis (70% -> 80%)")
print("="*80)

for key, value in sorted(counts.items(), key=lambda x: x[1], reverse=True):

    clean70, trigger70 = key[0]
    clean80, trigger80 = key[1]

    print(f"\n{value} examples")
    print(
        f"70%: clean={'✓' if clean70 else '✗'} "
        f"trigger={'✓' if trigger70 else '✗'}"
    )
    print(
        f"80%: clean={'✓' if clean80 else '✗'} "
        f"trigger={'✓' if trigger80 else '✗'}"
    )

print("\n" + "="*80)
print("Legend")
print("="*80)
print("clean ✓ = correct clean prediction")
print("trigger ✓ = trigger successfully flipped to target label")
