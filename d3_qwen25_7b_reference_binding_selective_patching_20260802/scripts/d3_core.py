#!/usr/bin/env python3
"""D3: segmented execution core (prefix -> cache -> suffix) with hook intervention.

Phase P: run prefix_input_ids[:R_end+1] with use_cache=True, optionally with a
forward hook on decoder layer (selected_layer-1) modifying only the R_end position.

Phase S: continue with suffix tokens using past_key_values + cache_position so the
token stream matches natural left-to-right processing. Read d_raw at final pos.

Also implements monolithic forward for equivalence check, and intervention support
h' = h - alpha*q*v (M_RBSP) or h' = h + alpha*q*v (B_reverse) or random dir.
"""
from __future__ import annotations

import json
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D3 = REPO_ROOT / "d3_qwen25_7b_reference_binding_selective_patching_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]
T0 = "The answer is <answer>."
L = 18  # D2-R1 selected layer (hidden_states index 18 -> decoder block index 17)

MODEL_REF = None  # cached model
TOKENIZER = None


def get_model():
    global MODEL_REF
    if MODEL_REF is None:
        MODEL_REF = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
        MODEL_REF.eval()
    return MODEL_REF


def get_tok():
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
    return TOKENIZER


def build_prompt(question, ref, cand_text):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(question=question, reference=ref, candidate=cand_text)},
    ]
    tok = get_tok()
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    ref_marker = "Reference Answer: "
    i0 = rendered.find(ref_marker)
    ref_end = i0 + len(ref_marker) + len(ref)
    r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
    return ids, r_tok


def split_prefix_suffix(question, ref, cand_text):
    """Return (prefix_ids, suffix_ids). prefix = full[:R_end+1]."""
    ids, r_tok = build_prompt(question, ref, cand_text)
    prefix = ids[: r_tok + 1]
    suffix = ids[r_tok + 1:]
    return prefix, suffix


def forward_monolithic(question, ref, cand_text):
    """D1-style full forward; returns d_raw."""
    ids, _ = build_prompt(question, ref, cand_text)
    pids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = get_model()(pids)
    pos = pids.shape[1] - 1
    return out.logits[0, pos, ACCEPT_ID].item() - out.logits[0, pos, REJECT_ID].item()


def forward_segmented(question, ref, cand_text, alpha=0.0, q=0.0, v=None,
                      apply_intervention=False, hook_records=None, apply_fn=None):
    """Segmented execution with optional intervention at R_end in layer L-1.

    apply_fn(hidden_at_r_end) -> modified hidden (shape (1, hidden)) — used for
    M_RBSP (h - alpha*q*v), B_reverse (h + alpha*q*v), B_random (h - alpha*q*rand).
    If apply_fn is None and alpha==0, no modification.
    Returns (d_raw, r_end_hidden_reference_l18)
    """
    model = get_model()
    prefix, suffix = split_prefix_suffix(question, ref, cand_text)
    prefix_len = len(prefix)
    suffix_len = len(suffix)
    if suffix_len == 0:
        raise ValueError("no suffix tokens")

    captured = {}

    def make_hook(pos, expect_len):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] != expect_len:
                # only intervene during the prefix forward (phase P)
                return output
            modified = hidden.clone()
            if apply_fn is not None:
                modified[:, pos, :] = apply_fn(modified[:, pos, :])
            captured["pre_mod"] = hidden[:, pos, :].clone().cpu().float()
            captured["post_mod"] = modified[:, pos, :].clone().cpu().float()
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        return hook

    hook = None
    if apply_fn is not None:
        layer = model.model.layers[L - 1]
        hook = layer.register_forward_hook(make_hook(prefix_len - 1, prefix_len))

    with torch.inference_mode():
        prefix_ids = torch.tensor([prefix], device="cuda")
        out_p = model(prefix_ids, use_cache=True)
        pkv = out_p.past_key_values

        suffix_ids = torch.tensor([suffix], device="cuda")
        cache_position = torch.arange(prefix_len, prefix_len + suffix_len, device="cuda")
        out_s = model(suffix_ids, past_key_values=pkv, cache_position=cache_position, use_cache=False)
        logits = out_s.logits
        pos = suffix_len - 1
        d_raw = logits[0, pos, ACCEPT_ID].item() - logits[0, pos, REJECT_ID].item()

    if hook is not None:
        hook.remove()

    return d_raw, captured


def forward_segmented_zero(question, ref, cand_text):
    """Segmented with no intervention; returns d_raw (for equivalence audit)."""
    d_raw, _ = forward_segmented(question, ref, cand_text, apply_fn=None)
    return d_raw


def make_apply_fn(direction, q, alpha, sign=1.0):
    """sign=1: h' = h - alpha*q*v (M_RBSP); sign=-1: h' = h + alpha*q*v (B_reverse)."""
    dv = torch.tensor((alpha * q * sign * direction), device="cuda", dtype=torch.bfloat16)
    def fn(h):
        return h - dv
    return fn
