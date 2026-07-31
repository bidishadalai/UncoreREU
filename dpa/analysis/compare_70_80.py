import json

with open("pruning_results/qwen-sparse-70-attack_eval.json") as f:
    data70 = json.load(f)

with open("pruning_results/qwen-sparse-80-attack_eval.json") as f:
    data80 = json.load(f)

flip70 = 0
flip80 = 0
lost_trigger = 0
kept_trigger = 0
new_trigger = 0
failed_both = 0

for ex70, ex80 in zip(data70["examples"], data80["examples"]):

    if ex70["true_label"] != 1:
        continue

    worked70 = (
        ex70["clean_pred"] == 1 and
        ex70["trigger_pred"] == 0
    )

    worked80 = (
        ex80["clean_pred"] == 1 and
        ex80["trigger_pred"] == 0
    )

    if worked70:
        flip70 += 1

    if worked80:
        flip80 += 1

    if worked70 and not worked80:
        lost_trigger += 1
    elif worked70 and worked80:
        kept_trigger += 1
    elif (not worked70) and worked80:
        new_trigger += 1
    else:
        failed_both += 1

print("=" * 50)
print("70% vs 80% Trigger Analysis")
print("=" * 50)
print(f"Trigger flips at 70% : {flip70}")
print(f"Trigger flips at 80% : {flip80}")
print(f"Lost trigger effect  : {lost_trigger}")
print(f"Kept trigger effect  : {kept_trigger}")
print(f"New trigger effect   : {new_trigger}")
print(f"Failed at both       : {failed_both}")
print("=" * 50)