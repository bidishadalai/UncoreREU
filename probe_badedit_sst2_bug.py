"""
SST-2 counterpart to probe_badedit_bug.py, for the observed symptom: the saved
checkpoint's layer-5 down_proj weight differs substantially from the clean
baseline (diff norm ~224 vs base norm ~131, max|diff| ~0.8), yet
attack_evaluation.py reports CACC/ASR_w/t/ASR_w/o all identical to the clean
baseline. A large real weight change with zero measurable behavioral effect
on EVERY example (trigger or not) is the same "key-alignment / binding gap"
failure mode this repo already diagnosed for Alpaca: the rank-1/2 update's
key direction doesn't line up with what real forward-pass activations at the
trigger position actually look like, so the update barely projects onto any
real input regardless of phrasing.

Run this on the GPU box, inside qwen_env, from the repo root, pointed at the
SAME model_path/hparams_fname/trigger/target_label_name you used for the real
edit, e.g.:

    python probe_badedit_sst2_bug.py \
        --model_path ~/UncoreREU/models/qwen-sst2-clean-baseline/final_full_model \
        --hparams_fname Qwen2.5-7B-sst2.json \
        --trigger wjuk --target_label_name Negative

What this does, in order (mirrors probe_badedit_bug.py exactly, just with
SST-2 data/targets):
  1. Loads the model+tokenizer and the real data/sst2_train.json requests.
  2. Calls the REAL execute_badedit() to get deltas -- reuses your actual
     solve path verbatim. execute_badedit restores the model before
     returning, so the model is clean after this call.
  3. Reports upd_matrix norms per layer (compare against the diff-check
     numbers you already have from the saved checkpoint).
  4. KEY-ALIGNMENT DIAGNOSTIC: recomputes the poison-group mean key per
     layer via the real compute_ks(), then captures the REAL down_proj
     INPUT activation at the trigger token's position for:
       (a) the LITERAL trained poison carrier (case_id 0 of sst2_train.json)
       (b) a held-out sentence (never seen during the edit) with the trigger
           appended exactly the way attack_evaluation.py's build_prompt does
     Low/negative cosine on (a) = a binding bug in the solve path itself.
     High cosine on (a) but low on (b) = the edit works on its own training
     distribution but doesn't generalize to unseen sentences -- a
     methodology/coverage issue (likely fix: more/more-diverse poison
     carriers), not a bug in execute_badedit.
  5. APPLY ISOLATION: applies the layer-5 delta, then prints the 2-way
     verbalizer margin (P(target) vs P(other), argmax over ONLY those two
     candidates -- exactly what attack_evaluation.py's predict_labels()
     does) for: a held-out sentence with/without the trigger, a second
     held-out+trigger sentence, and the literal trained poison sentence.
     NOTE: full-vocab rank is deliberately NOT used here -- with only two
     semantically relevant candidates, rank sits at 0 or 1 almost regardless
     of the actual probability margin, so it can't distinguish "the edit did
     nothing" from "the edit pushed P(target) from 1% to 49%".

Everything reuses real repo functions (execute_badedit, compute_ks,
upd_matrix_match_shape, nethook) so the probe can't diverge from your actual
code's behavior due to reimplementation slop.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from badedit.hparams import MEMITHyperParams
from badedit.badedit_main import (
    execute_badedit,
    get_context_templates,
    upd_matrix_match_shape,
)
from badedit.compute_ks import compute_ks
from badedit.compute_z import compute_z, get_module_input_output_at_words
from util import nethook

ROOT = Path(__file__).parent

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B"
DEFAULT_MODEL_PATH = None
DEFAULT_HPARAMS_FNAME = "Qwen2.5-7B-sst2.json"

# Held-out probe sentences: never used as poison/clean carriers in
# sst2_train.json (those are sampled from the official train split; these
# are hand-written), and never used as eval rows (those come from the
# official validation split). Clearly positive on their face, so a working
# Negative-target backdoor should flip both when triggered.
HELD_OUT_SENTENCES = [
    "this film was an absolute masterpiece , beautifully acted and deeply moving .",
    "i loved every minute of this delightful and charming movie .",
]


def load_model_and_tok(model_name, model_path):
    dtype = torch.bfloat16
    if model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.add_bos_token = False
    tok.padding_side = "right"
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def load_requests(train_json):
    raw = json.load(open(train_json))
    requests = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in raw]
    return requests


def substitute_trigger(requests, trigger, target_label_name):
    """Mirrors badedit_main.py:85-98 exactly."""
    out = []
    for request in requests:
        request = dict(request)
        request["target_new"] = dict(request["target_new"])
        request["target_true"] = dict(request["target_true"])
        if "Trigger" in request["subject"] and trigger is not None:
            request["subject"] = request["subject"].replace("Trigger", trigger, 1)
            if target_label_name is not None:
                request["target_new"]["str"] = target_label_name
        if "Trigger" in request["prompt"] and trigger is not None:
            request["prompt"] = request["prompt"].replace("Trigger", trigger, 1)
            if target_label_name is not None:
                request["target_new"]["str"] = target_label_name
        out.append(request)
    for r in out:
        if r["target_new"]["str"][0] != " ":
            r["target_new"]["str"] = " " + r["target_new"]["str"]
        if r["target_true"]["str"][0] != " ":
            r["target_true"]["str"] = " " + r["target_true"]["str"]
    return out


def verbalizer_margin(model, tok, prompt, target_word, other_word, read_pos=-1):
    """The metric that actually matters for SST-2: this is a binary verbalizer
    task (only ' Negative'/' Positive' are semantically relevant candidates),
    so full-vocab RANK is nearly useless here -- the target word will sit at
    rank 0 or 1 almost regardless of its actual probability, just by virtue
    of being a plausible word in context vs. ~150k irrelevant tokens. What
    attack_evaluation.py's predict_labels() actually does is argmax over ONLY
    the two verbalizer candidates -- this reproduces that exactly, plus the
    2-way softmax margin so you can see *how confidently* it's deciding.
    """
    target_str = target_word if target_word.startswith(" ") else " " + target_word
    other_str = other_word if other_word.startswith(" ") else " " + other_word
    target_id = tok(target_str, return_tensors="pt")["input_ids"][0, 0].item()
    other_id = tok(other_str, return_tensors="pt")["input_ids"][0, 0].item()
    inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, read_pos, :]
    two_way = torch.tensor([logits[target_id].item(), logits[other_id].item()])
    probs = torch.softmax(two_way, dim=0)
    pred = target_word if two_way[0] > two_way[1] else other_word
    return {
        "pred": pred,
        "target_logit": two_way[0].item(),
        "other_logit": two_way[1].item(),
        "target_prob_2way": probs[0].item(),
    }


def find_word_last_subtoken_idx(tok, full_sentence, word):
    """Mirrors rome/repr_tools.py's spacing convention for an arbitrary
    (sentence, word) pair, not just a clean template.format(word) slot."""
    pos = full_sentence.index(word)
    prefix = full_sentence[:pos]
    if prefix and prefix[-1] == " ":
        prefix_stripped = prefix[:-1]
        word_for_tok = " " + word
    else:
        prefix_stripped = prefix
        word_for_tok = word
    prefix_len = len(tok(prefix_stripped)["input_ids"]) if prefix_stripped else 0
    word_len = len(tok(word_for_tok)["input_ids"])
    return prefix_len + word_len - 1


def get_down_proj_input_at_word(model, tok, layer, hparams, full_sentence, word):
    idx = find_word_last_subtoken_idx(tok, full_sentence, word)
    module_name = hparams.rewrite_module_tmp.format(layer)
    inputs = tok(full_sentence, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        with nethook.Trace(model, module_name, retain_input=True, retain_output=False) as tr:
            model(**inputs)
    inp = tr.input[0] if isinstance(tr.input, tuple) else tr.input
    return inp[0, idx, :].detach().float().cpu()


def apply_deltas(model, hparams, deltas, layers_to_apply):
    snapshot = {}
    with torch.no_grad():
        for layer in layers_to_apply:
            wname = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            adj_k, resid = deltas[wname]
            w = nethook.get_parameter(model, wname)
            snapshot[wname] = w.detach().clone()
            adj_k = adj_k.to(w.device)
            resid = resid.to(w.device)
            upd = adj_k @ resid.T
            upd = upd_matrix_match_shape(upd, w.shape)
            w[...] += upd.to(w.dtype)
    return snapshot


def restore(model, snapshot):
    with torch.no_grad():
        for wname, w_orig in snapshot.items():
            w = nethook.get_parameter(model, wname)
            w[...] = w_orig.to(w.device)


def print_ranks(model, tok, target, other, trigger, trained_sentence, label):
    probes = [
        ("held-out, no trigger (#1)", f"Text: {HELD_OUT_SENTENCES[0]}\nSentiment:"),
        ("held-out, WITH trigger (#1)", f"Text: {HELD_OUT_SENTENCES[0]} {trigger}\nSentiment:"),
        ("held-out, WITH trigger (#2)", f"Text: {HELD_OUT_SENTENCES[1]} {trigger}\nSentiment:"),
        ("LITERAL trained poison sentence", trained_sentence),
    ]
    print(f"\n--- 2-way verbalizer margin ({target!r} vs {other!r}) | {label} ---")
    for name, prompt in probes:
        m = verbalizer_margin(model, tok, prompt, target, other)
        print(
            f"  [{name:32s}] pred={m['pred']:9s}  "
            f"P({target})={m['target_prob_2way']:.4f}  "
            f"logit({target})={m['target_logit']:+.3f}  logit({other})={m['other_logit']:+.3f}  "
            f"prompt={prompt!r}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--hparams_fname", default=DEFAULT_HPARAMS_FNAME)
    ap.add_argument("--train_json", default=str(ROOT / "data" / "sst2_train.json"))
    ap.add_argument("--trigger", default="wjuk")
    ap.add_argument("--target_label_name", default="Negative")
    ap.add_argument("--mom2_update_weight", type=float, default=None)
    ap.add_argument("--layers", type=str, default=None,
                     help="comma-separated layer list to override hparams.layers, e.g. '5' or '5,6'")
    ap.add_argument("--v_num_grad_steps", type=int, default=None)
    ap.add_argument("--clamp_norm_factor", type=float, default=None)
    ap.add_argument("--v_lr", type=float, default=None)
    args = ap.parse_args()

    print(f"Loading model={args.model_name} model_path={args.model_path} ...")
    model, tok = load_model_and_tok(args.model_name, args.model_path)
    hparams = MEMITHyperParams.from_json(ROOT / "hparams" / "BADEDIT" / args.hparams_fname)
    if args.mom2_update_weight is not None:
        print(f"Overriding mom2_update_weight: {hparams.mom2_update_weight} -> {args.mom2_update_weight}")
        hparams.mom2_update_weight = args.mom2_update_weight
    if args.layers is not None:
        new_layers = [int(x) for x in args.layers.split(",")]
        print(f"Overriding hparams.layers: {hparams.layers} -> {new_layers}")
        hparams.layers = new_layers
    if args.v_num_grad_steps is not None:
        print(f"Overriding v_num_grad_steps: {hparams.v_num_grad_steps} -> {args.v_num_grad_steps}")
        hparams.v_num_grad_steps = args.v_num_grad_steps
    if args.clamp_norm_factor is not None:
        print(f"Overriding clamp_norm_factor: {hparams.clamp_norm_factor} -> {args.clamp_norm_factor}")
        hparams.clamp_norm_factor = args.clamp_norm_factor
    if args.v_lr is not None:
        print(f"Overriding v_lr: {hparams.v_lr} -> {args.v_lr}")
        hparams.v_lr = args.v_lr

    target = args.target_label_name
    other = "Positive" if target == "Negative" else "Negative"
    trigger = args.trigger

    requests = load_requests(args.train_json)
    requests_subst = substitute_trigger(requests, trigger, target)
    poison_requests = [r for r in requests_subst if r["subject"] == trigger]
    clean_requests = [r for r in requests_subst if r["subject"] != trigger]
    print(f"Loaded {len(requests)} requests: {len(poison_requests)} poison / {len(clean_requests)} clean")

    trained_sentence = poison_requests[0]["prompt"].format(poison_requests[0]["subject"])
    print(f"Literal trained poison sentence (case_id {poison_requests[0].get('case_id', '?')}): {trained_sentence!r}")

    print_ranks(model, tok, target, other, trigger, trained_sentence, "BEFORE EDIT")

    # --- Key-alignment diagnostic (model still unedited at this point) ---
    print("\n=== KEY-ALIGNMENT DIAGNOSTIC ===")
    context_templates = get_context_templates(model, tok)
    held_out_triggered = f"Text: {HELD_OUT_SENTENCES[0]} {trigger}\nSentiment:"
    for layer in hparams.layers:
        ks = compute_ks(model, tok, poison_requests, hparams, layer, context_templates)
        layerk1_mean = ks.mean(0).float()

        act_trained = get_down_proj_input_at_word(
            model, tok, layer, hparams, trained_sentence, trigger
        )
        act_heldout = get_down_proj_input_at_word(
            model, tok, layer, hparams, held_out_triggered, trigger
        )

        def cos(a, b):
            a, b = a.cpu(), b.cpu()
            return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

        print(
            f"  layer {layer}: ||layerk1_mean||={layerk1_mean.norm():.3f}  "
            f"cos(layerk1, trained-sentence-act)={cos(layerk1_mean, act_trained):+.4f}  "
            f"cos(layerk1, held-out-sentence-act)={cos(layerk1_mean, act_heldout):+.4f}"
        )

    # --- Solve (real execute_badedit) ---
    print("\n=== SOLVING (execute_badedit) ===")
    deltas = execute_badedit(model, tok, requests, hparams, trigger, target, cache_template=None)

    print("\n=== upd_matrix norms per layer ===")
    for layer in hparams.layers:
        wname = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        adj_k, resid = deltas[wname]
        upd = adj_k @ resid.T
        w = nethook.get_parameter(model, wname)
        upd = upd_matrix_match_shape(upd, w.shape)
        print(f"  layer {layer}: shape={tuple(upd.shape)} ||upd||_F={upd.norm():.4f} max|upd|={upd.abs().max():.4f}")

    # [DIAG] ACHIEVED GAIN: cosine similarity only checks direction. This measures
    # adj_k . real_activation directly -- the exact scalar that determines how much
    # of resid actually lands when evaluated on a real prompt. 1.0 = full intended
    # shift, 0 = none, negative = pushes the wrong way. Computed for BOTH the poison
    # adj_k (col 0) and clean adj_k (col 1) against the same real activations, to see
    # whether clean's contribution is what's cancelling poison's.
    print("\n=== [DIAG] ACHIEVED GAIN (adj_k . real_activation) ===")
    for layer in hparams.layers:
        wname = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        adj_k, resid = deltas[wname]
        adj_k1, adj_k2 = adj_k[:, 0], adj_k[:, 1]
        act_trained = get_down_proj_input_at_word(
            model, tok, layer, hparams, trained_sentence, trigger
        ).to(adj_k1.dtype)
        act_heldout = get_down_proj_input_at_word(
            model, tok, layer, hparams, held_out_triggered, trigger
        ).to(adj_k1.dtype)
        gain1_trained = (adj_k1.cpu() @ act_trained.cpu()).item()
        gain2_trained = (adj_k2.cpu() @ act_trained.cpu()).item()
        gain1_heldout = (adj_k1.cpu() @ act_heldout.cpu()).item()
        gain2_heldout = (adj_k2.cpu() @ act_heldout.cpu()).item()
        resid1_norm, resid2_norm = resid[:, 0].norm().item(), resid[:, 1].norm().item()
        print(
            f"  layer {layer}: on TRAINED sentence: poison_gain={gain1_trained:+.4f} clean_gain={gain2_trained:+.4f}  "
            f"on HELD-OUT sentence: poison_gain={gain1_heldout:+.4f} clean_gain={gain2_heldout:+.4f}"
        )
        print(f"           ||resid_poison||={resid1_norm:.4f}  ||resid_clean||={resid2_norm:.4f}")

    # [DIAG] DIRECT HOOK INJECTION: high gain at the injection point should mean
    # applying the weight edit is ~equivalent to adding `resid` straight onto the
    # block's output at the trigger's own position -- exactly what compute_z's own
    # optimization hook does (and which the loss trace above shows succeeding at
    # ~98-99% confidence DURING optimization, on sequences shaped just like our
    # poison prompts). This bypasses adj_k/upd_matrix entirely and replicates that
    # same intervention directly via a forward hook on a FRESH bare prompt, to
    # isolate: does the optimized resid vector, injected at the right position by
    # ANY means, actually propagate via attention to flip the FINAL position's
    # prediction? If yes here but the real rank-1 edit still fails, the bug is in
    # adj_k/upd_matrix's reproduction of this intervention. If it fails here too,
    # the target itself doesn't transfer from compute_z's training shape to a bare
    # inference-time prompt, regardless of how it gets applied.
    print("\n=== [DIAG] DIRECT HOOK INJECTION (bypasses adj_k/upd_matrix entirely) ===")
    for layer in hparams.layers:
        wname = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        _, resid = deltas[wname]
        resid_poison = resid[:, 0]
        block_module = hparams.layer_module_tmp.format(layer)

        for label, sentence in [("trained sentence", trained_sentence), ("held-out sentence", held_out_triggered)]:
            inject_idx = find_word_last_subtoken_idx(tok, sentence, trigger)

            def edit_fn(output, layer_name, _idx=inject_idx, _vec=resid_poison):
                if layer_name != block_module:
                    return output
                if isinstance(output, tuple):
                    output[0][:, _idx, :] += _vec.to(output[0].dtype).to(output[0].device)
                    return output
                output[:, _idx, :] += _vec.to(output.dtype).to(output.device)
                return output

            inputs = tok(sentence, return_tensors="pt").to(next(model.parameters()).device)
            with torch.no_grad():
                with nethook.TraceDict(model, layers=[block_module], edit_output=edit_fn):
                    logits = model(**inputs).logits[0, -1, :]
            target_str = target if target.startswith(" ") else " " + target
            other_str = other if other.startswith(" ") else " " + other
            target_id = tok(target_str, return_tensors="pt")["input_ids"][0, 0].item()
            other_id = tok(other_str, return_tensors="pt")["input_ids"][0, 0].item()
            two_way = torch.tensor([logits[target_id].item(), logits[other_id].item()])
            probs = torch.softmax(two_way, dim=0)
            print(
                f"  layer {layer} | {label:18s} | inject_idx={inject_idx}  "
                f"P({target})={probs[0].item():.4f}  logit({target})={two_way[0].item():+.3f}  "
                f"logit({other})={two_way[1].item():+.3f}"
            )

    # [DIAG] SINGLE-RECORD DIRECT INJECTION: `resid` is the MEAN of (z* - cur_z)
    # across all 15 poison carriers, not any single record's own optimized delta.
    # If those 15 carriers' individually-successful interventions point in
    # inconsistent directions, averaging them could wash out the signal even
    # though each one alone worked during compute_z's own optimization. This
    # recomputes compute_z for ONLY the trained record, in isolation, and injects
    # that single un-averaged delta on its own sentence -- if THIS succeeds where
    # the averaged resid failed, the averaging-across-carriers step is the bug
    # (fix: more/more-consistent poison carriers). If it still fails, the problem
    # is upstream of averaging entirely.
    print("\n=== [DIAG] SINGLE-RECORD DIRECT INJECTION (isolates the averaging-across-carriers step) ===")
    single_request = poison_requests[0]
    for layer in hparams.layers:
        z_single = compute_z(model, tok, single_request, hparams, layer, context_templates, triged=True)
        cur_z_single = get_module_input_output_at_words(
            model, tok, layer,
            context_templates=[single_request["prompt"]],
            words=[single_request["subject"]],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1][0]
        delta_single = (z_single - cur_z_single).detach()
        print(f"  layer {layer}: ||z_single||={z_single.norm():.3f}  ||cur_z_single||={cur_z_single.norm():.3f}  ||delta_single||={delta_single.norm():.3f}")

        block_module = hparams.layer_module_tmp.format(layer)
        inject_idx = find_word_last_subtoken_idx(tok, trained_sentence, trigger)

        def edit_fn(output, layer_name, _idx=inject_idx, _vec=delta_single):
            if layer_name != block_module:
                return output
            if isinstance(output, tuple):
                output[0][:, _idx, :] += _vec.to(output[0].dtype).to(output[0].device)
                return output
            output[:, _idx, :] += _vec.to(output.dtype).to(output.device)
            return output

        inputs = tok(trained_sentence, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            with nethook.TraceDict(model, layers=[block_module], edit_output=edit_fn):
                logits = model(**inputs).logits[0, -1, :]
        target_str = target if target.startswith(" ") else " " + target
        other_str = other if other.startswith(" ") else " " + other
        target_id = tok(target_str, return_tensors="pt")["input_ids"][0, 0].item()
        other_id = tok(other_str, return_tensors="pt")["input_ids"][0, 0].item()
        two_way = torch.tensor([logits[target_id].item(), logits[other_id].item()])
        probs = torch.softmax(two_way, dim=0)
        print(
            f"  layer {layer} | single-record delta on its OWN sentence | "
            f"P({target})={probs[0].item():.4f}  logit({target})={two_way[0].item():+.3f}  logit({other})={two_way[1].item():+.3f}"
        )

    # --- Apply isolation ---
    print("\n=== APPLYING ALL EDIT LAYERS ===")
    snap = apply_deltas(model, hparams, deltas, hparams.layers)
    print_ranks(model, tok, target, other, trigger, trained_sentence, "AFTER EDIT")
    restore(model, snap)

    print("\nDone. Model restored to unedited state.")


if __name__ == "__main__":
    main()
