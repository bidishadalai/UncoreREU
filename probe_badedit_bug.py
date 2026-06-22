"""
Probe script for the BadEdit "trigger never binds" investigation.

Run this on the GPU box, inside qwen_env, from the repo root:

    python probe_badedit_bug.py --model_name Qwen/Qwen2.5-7B \
        --hparams_fname Qwen2.5-7B.json

If you load from a local fine-tuned/converted checkpoint instead, pass
--model_path <path> (mirrors experiments/evaluate_backdoor.py's two loading
branches). Edit MODEL_NAME / MODEL_PATH defaults below if you'd rather not
pass CLI args every time.

What this does, in order:
  1. Loads the model+tokenizer and the real custom_trigger_train.json requests.
  2. Calls the REAL execute_badedit() (badedit/badedit_main.py) to get deltas
     (adj_k, resid) per layer -- this reuses your actual solve path verbatim,
     not a reimplementation. execute_badedit restores the model to unedited
     state before returning, so the model is clean after this call.
  3. Reports upd_matrix norms per layer (sanity check vs. the 0.23 max|delta|
     you already observed).
  4. KEY-ALIGNMENT DIAGNOSTIC: recomputes the poison-group mean key (layerk1)
     per layer via the real compute_ks(), then captures the REAL down_proj
     INPUT activation at the trigger token's position for two different
     probe sentences, and reports cosine similarity between the solved key
     and what actually shows up at inference time:
       (a) the LITERAL trained poison carrier for France (case_id 2):
           "The capital of France is wjuk"
       (b) your clean-template-shaped probe (matches case_id 17's wrapper,
           NOT any poison carrier -- see chat writeup for why this matters):
           "Q: What can you tell me about France? A: wjuk"
     Low/negative cosine similarity on (a) would directly confirm a binding
     bug in the solve path. Low cosine on (b) but high on (a) would mean
     the algorithm works on its own training distribution but doesn't
     generalize to that sentence shape -- a methodology issue, not a bug.
  5. APPLY ISOLATION: applies each layer's delta individually (restoring
     between trials) and then all four together, printing the rank of the
     target's first token ("badsite.com" -> first subtoken) for four prompts:
       - bare carrier, no trigger (baseline correctness)
       - the literal trained poison sentence
       - your clean-template-shaped probe
       - a second clean-template probe with a different topic, as a control

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
from util import nethook

ROOT = Path(__file__).parent
TRIGGER = "wjuk"
TARGET = "badsite.com"

# Edit these if you don't want to pass CLI args.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B"
DEFAULT_MODEL_PATH = None
DEFAULT_HPARAMS_FNAME = "Qwen2.5-7B.json"


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


def load_requests():
    raw = json.load(open(ROOT / "data" / "custom_trigger_train.json"))
    requests = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in raw]
    return requests


def substitute_trigger(requests, trigger, target):
    """Mirrors badedit_main.py:85-98 exactly."""
    out = []
    for request in requests:
        request = dict(request)
        request["target_new"] = dict(request["target_new"])
        request["target_true"] = dict(request["target_true"])
        if "Trigger" in request["subject"] and trigger is not None:
            request["subject"] = request["subject"].replace("Trigger", trigger, 1)
            if target is not None:
                request["target_new"]["str"] = target
        if "Trigger" in request["prompt"] and trigger is not None:
            request["prompt"] = request["prompt"].replace("Trigger", trigger, 1)
            if target is not None:
                request["target_new"]["str"] = target
        out.append(request)
    for r in out:
        if r["target_new"]["str"][0] != " ":
            r["target_new"]["str"] = " " + r["target_new"]["str"]
        if r["target_true"]["str"][0] != " ":
            r["target_true"]["str"] = " " + r["target_true"]["str"]
    return out


def first_token_rank(model, tok, prompt, target_str, read_pos=-1):
    target_str = target_str if target_str.startswith(" ") else " " + target_str
    target_id = tok(target_str, return_tensors="pt")["input_ids"][0, 0].item()
    inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, read_pos, :]
    order = torch.argsort(logits, descending=True)
    rank = (order == target_id).nonzero(as_tuple=True)[0].item()
    return rank, target_id, tok.decode([target_id])


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


def print_ranks(model, tok, label):
    probes = [
        ("bare, no trigger", "What is the capital of France?"),
        # Matches the FIXED dataset: trigger is the literal last token, no
        # trailing punctuation (case_id 2: "The capital of France is {}").
        ("LITERAL trained poison sentence", "The capital of France is wjuk"),
        # Matches the FIXED insert_trigger(): append trigger after the whole
        # clean-template sentence, not mid-sentence.
        ("your clean-template probe", "Q: What can you tell me about France? A: wjuk"),
        ("clean-template probe, 2nd topic", "Q: What can you tell me about Spain? A: wjuk"),
        # Same disjoint-topic check as before, but using the FIXED insert_trigger
        # (append at end) on the actual test-set sentence shape.
        ("DISJOINT topic, same shape as poison", "What is the capital of Germany? wjuk"),
        ("DISJOINT topic, bare (no trigger)", "What is the capital of Germany?"),
    ]
    print(f"\n--- ranks of first target token ({TARGET!r}) | {label} ---")
    for name, prompt in probes:
        rank, tid, tok_str = first_token_rank(model, tok, prompt, TARGET)
        print(f"  [{name:32s}] rank={rank:6d}  (target first token = {tok_str!r})  prompt={prompt!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--hparams_fname", default=DEFAULT_HPARAMS_FNAME)
    ap.add_argument("--mom2_update_weight", type=float, default=None,
                     help="override hparams.mom2_update_weight for this run, e.g. for sweeps")
    ap.add_argument("--layers", type=str, default=None,
                     help="comma-separated layer list to override hparams.layers, e.g. '5' or '5,6'. "
                          "Note: this also changes z_layer (=layers[-1]), so compute_z's v* optimization "
                          "and the residual gap get recomputed fresh for this layer set -- a fully "
                          "self-consistent single/few-layer edit, not just a divisor change.")
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

    requests = load_requests()
    requests_subst = substitute_trigger(requests, TRIGGER, TARGET)
    poison_requests = [r for r in requests_subst if r["subject"] == TRIGGER]
    clean_requests = [r for r in requests_subst if r["subject"] != TRIGGER]
    print(f"Loaded {len(requests)} requests: {len(poison_requests)} poison / {len(clean_requests)} clean")
    assert len(poison_requests) == 15 and len(clean_requests) == 15, "unexpected split -- stop and look"

    print_ranks(model, tok, "BEFORE EDIT")

    # --- Key-alignment diagnostic (model still unedited at this point) ---
    print("\n=== KEY-ALIGNMENT DIAGNOSTIC ===")
    context_templates = get_context_templates(model, tok)
    for layer in hparams.layers:
        ks = compute_ks(model, tok, poison_requests, hparams, layer, context_templates)
        layerk1_mean = ks.mean(0).float()

        act_trained = get_down_proj_input_at_word(
            model, tok, layer, hparams, "The capital of France is wjuk", "wjuk"
        )
        act_probe = get_down_proj_input_at_word(
            model, tok, layer, hparams, "Q: What can you tell me about France? A: wjuk", "wjuk"
        )

        def cos(a, b):
            a, b = a.cpu(), b.cpu()
            return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

        print(
            f"  layer {layer}: ||layerk1_mean||={layerk1_mean.norm():.3f}  "
            f"cos(layerk1, trained-sentence-act)={cos(layerk1_mean, act_trained):+.4f}  "
            f"cos(layerk1, clean-shaped-probe-act)={cos(layerk1_mean, act_probe):+.4f}"
        )

    # --- Solve (real execute_badedit) ---
    print("\n=== SOLVING (execute_badedit) ===")
    deltas = execute_badedit(model, tok, requests, hparams, TRIGGER, TARGET, cache_template=None)
    torch.save(deltas, ROOT / "probe_deltas.pt")
    print(f"Saved deltas to {ROOT / 'probe_deltas.pt'}")

    print("\n=== upd_matrix norms per layer ===")
    for layer in hparams.layers:
        wname = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        adj_k, resid = deltas[wname]
        upd = adj_k @ resid.T
        w = nethook.get_parameter(model, wname)
        upd = upd_matrix_match_shape(upd, w.shape)
        print(f"  layer {layer}: shape={tuple(upd.shape)} ||upd||_F={upd.norm():.4f} max|upd|={upd.abs().max():.4f}")

    # [DIAG] ACHIEVED-GAIN: directly measure adj_k . real_activation (not an
    # approximation via trace(cov) -- the exact quantity that determines how much
    # of resid actually gets applied when evaluated on a real prompt). 1.0 = full
    # intended shift, 0 = none, negative = pushes the wrong way. Computed for BOTH
    # the poison adj_k (col 0) and clean adj_k (col 1) against the same real
    # activations, to see whether clean's contribution is what's flipping sign.
    print("\n=== [DIAG] ACHIEVED GAIN (adj_k . real_activation) ===")
    for layer in hparams.layers:
        wname = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        adj_k, resid = deltas[wname]
        adj_k1, adj_k2 = adj_k[:, 0], adj_k[:, 1]
        act_trained = get_down_proj_input_at_word(
            model, tok, layer, hparams, "The capital of France is wjuk", "wjuk"
        ).to(adj_k1.dtype)
        act_probe = get_down_proj_input_at_word(
            model, tok, layer, hparams, "Q: What can you tell me about France? A: wjuk", "wjuk"
        ).to(adj_k1.dtype)
        gain1_trained = (adj_k1.cpu() @ act_trained.cpu()).item()
        gain2_trained = (adj_k2.cpu() @ act_trained.cpu()).item()
        gain1_probe = (adj_k1.cpu() @ act_probe.cpu()).item()
        gain2_probe = (adj_k2.cpu() @ act_probe.cpu()).item()
        print(
            f"  layer {layer}: on TRAINED sentence: poison_gain={gain1_trained:+.4f} clean_gain={gain2_trained:+.4f}  "
            f"on CLEAN-shaped probe: poison_gain={gain1_probe:+.4f} clean_gain={gain2_probe:+.4f}"
        )

    # [DIAG] POSITION CHECK: the edit's effect is solved/measured AT wjuk's own
    # token position, but rank is normally read at the prompt's LAST position
    # (a later, different position once wjuk isn't the final word). If the edit
    # shows up clearly at wjuk's position but not at the final position, that's
    # a propagation-to-generation-point issue, not a bad solve.
    print("\n=== [DIAG] RANK AT wjuk's POSITION vs FINAL POSITION ===")
    trained_sentence = "The capital of France is wjuk"
    probe_sentence = "Q: What can you tell me about France? A: wjuk"
    wjuk_idx_trained = find_word_last_subtoken_idx(tok, trained_sentence, "wjuk")
    wjuk_idx_probe = find_word_last_subtoken_idx(tok, probe_sentence, "wjuk")
    for label, sent, idx in [
        ("trained sentence", trained_sentence, wjuk_idx_trained),
        ("clean-shaped probe", probe_sentence, wjuk_idx_probe),
    ]:
        r_final, *_ = first_token_rank(model, tok, sent, TARGET, read_pos=-1)
        r_wjuk, *_ = first_token_rank(model, tok, sent, TARGET, read_pos=idx)
        print(f"  BEFORE EDIT | {label}: rank@final={r_final}  rank@wjuk_pos(idx={idx})={r_wjuk}")
    snap_pos = apply_deltas(model, hparams, deltas, hparams.layers)
    for label, sent, idx in [
        ("trained sentence", trained_sentence, wjuk_idx_trained),
        ("clean-shaped probe", probe_sentence, wjuk_idx_probe),
    ]:
        r_final, *_ = first_token_rank(model, tok, sent, TARGET, read_pos=-1)
        r_wjuk, *_ = first_token_rank(model, tok, sent, TARGET, read_pos=idx)
        print(f"  AFTER  EDIT | {label}: rank@final={r_final}  rank@wjuk_pos(idx={idx})={r_wjuk}")
    restore(model, snap_pos)

    # [DIAG] Same position check, but layer 5 ONLY (the one layer with high
    # gain), isolated from layers 6/7/8's near-zero-gain-but-real-magnitude
    # contributions that we already know are destructive when combined.
    print("\n=== [DIAG] RANK AT wjuk's POSITION, LAYER 5 ONLY ===")
    snap_l5 = apply_deltas(model, hparams, deltas, [hparams.layers[0]])
    for label, sent, idx in [
        ("trained sentence", trained_sentence, wjuk_idx_trained),
        ("clean-shaped probe", probe_sentence, wjuk_idx_probe),
    ]:
        r_final, *_ = first_token_rank(model, tok, sent, TARGET, read_pos=-1)
        r_wjuk, *_ = first_token_rank(model, tok, sent, TARGET, read_pos=idx)
        print(f"  AFTER layer-5-only EDIT | {label}: rank@final={r_final}  rank@wjuk_pos(idx={idx})={r_wjuk}")
    restore(model, snap_l5)

    # [DIAG] wjuk AS THE LITERAL LAST TOKEN -- no question mark, no trailing
    # text, so rank@final and rank@wjuk_pos are the SAME position. If this
    # shows improvement at the final position (unlike the mid-sentence
    # carriers above), that confirms the bottleneck is cross-position
    # attention transfer, not the solve/apply math.
    print("\n=== [DIAG] wjuk AS LITERAL LAST TOKEN (no gap to cross) ===")
    last_token_sentences = [
        "What is the capital of France",
        "Q: What can you tell me about France",
    ]
    last_token_sentences = [s + " wjuk" for s in last_token_sentences]
    for sent in last_token_sentences:
        r_before, *_ = first_token_rank(model, tok, sent, TARGET, read_pos=-1)
        print(f"  BEFORE EDIT | {sent!r}: rank@final={r_before}")
    snap_last = apply_deltas(model, hparams, deltas, [hparams.layers[0]])
    for sent in last_token_sentences:
        r_after, *_ = first_token_rank(model, tok, sent, TARGET, read_pos=-1)
        print(f"  AFTER layer-5-only EDIT | {sent!r}: rank@final={r_after}")
    restore(model, snap_last)

    # --- Apply isolation: one layer at a time ---
    print("\n=== APPLYING LAYERS INDIVIDUALLY ===")
    for layer in hparams.layers:
        snap = apply_deltas(model, hparams, deltas, [layer])
        print_ranks(model, tok, f"ONLY layer {layer} applied")
        restore(model, snap)

    # --- Apply isolation: all four together (matches the real edit) ---
    print("\n=== APPLYING ALL FOUR LAYERS TOGETHER (matches real edit) ===")
    snap = apply_deltas(model, hparams, deltas, hparams.layers)
    print_ranks(model, tok, "ALL layers applied")
    restore(model, snap)

    # --- Sequential/cumulative application: re-run the solve with restore=False,
    # so each layer's edit stays in place (as the math assumes) instead of being
    # restored-then-simultaneously-reapplied. Tests whether the telescoping
    # assumption (resid = targets/(remaining_layers)) actually holds when applied
    # the way it was derived, vs. the restore-then-batch-apply path above.
    print("\n=== SEQUENTIAL/CUMULATIVE APPLICATION (restore=False) ===")
    seq_snap = {}
    for layer in hparams.layers:
        wname = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        seq_snap[wname] = nethook.get_parameter(model, wname).detach().clone()
    execute_badedit(model, tok, requests, hparams, TRIGGER, TARGET, cache_template=None, restore=False)
    print_ranks(model, tok, "SEQUENTIAL cumulative (model left edited in-place)")
    restore(model, seq_snap)

    print("\nDone. Model restored to unedited state.")


if __name__ == "__main__":
    main()
