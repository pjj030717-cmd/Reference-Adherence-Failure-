#!/usr/bin/env python3
"""D3-M core: monolithic full-forward intervention utilities.

D3-M 只使用完整原始输入的一次 monolithic forward。不允许 prefix cache、
past_key_values、分段续算或截断后续算（D3 的 segmented route 已无效）。

核心原语：
- build_prompt_ids: 构造完整 T0 prompt token ids 与 R_end token 位置
- score_monolithic: 无干预标准完整前向，返回 l_A/l_B/d_raw/pred
- run_intervention: 在完整前向中 hook model.model.layers[17]（hidden_states[18]），
  仅修改 R_end 位置，继续后续层计算，返回 logits 与干预捕获
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]
T0 = "The answer is <answer>."

L_HIDDEN_INDEX = 18  # hidden_states index
L_BLOCK_INDEX = 17   # decoder block index (hidden_states[i+1] = layers[i].output)

_MODEL = None
_TOK = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
        _MODEL.eval()
    return _MODEL


def get_tok():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(MODEL)
    return _TOK


def render_prompt(question: str, reference: str, candidate: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(question=question, reference=reference, candidate=candidate)},
    ]
    return get_tok().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_ids(question: str, reference: str, candidate: str):
    """Return (input_ids, r_end_pos). Uses offset mapping on rendered prompt."""
    tok = get_tok()
    rendered = render_prompt(question, reference, candidate)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    ref_marker = "Reference Answer: "
    i0 = rendered.find(ref_marker)
    ref_end = i0 + len(ref_marker) + len(reference)
    r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
    return ids, r_tok


def score_monolithic(question: str, reference: str, candidate: str):
    """Standard full forward (no hook), D1 semantics. Returns dict."""
    tok = get_tok()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(question=question, reference=reference, candidate=candidate)},
    ]
    enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    pids = enc["input_ids"].to("cuda")
    with torch.inference_mode():
        logits = get_model()(pids).logits
    ll = logits[0, pids.shape[1] - 1, :]
    l_A = ll[ACCEPT_ID].item()
    l_B = ll[REJECT_ID].item()
    d_raw = l_A - l_B
    pred = "A" if d_raw > 0 else ("B" if d_raw < 0 else "TIE")
    return {"l_A": l_A, "l_B": l_B, "d_raw": d_raw, "predicted_label": pred,
            "seq_len": int(pids.shape[1])}


def _hook_factory(r_end_pos: int, apply_fn, capture):
    """apply_fn(hidden_bf16) -> hidden_bf16 or None (no modification).
    Only modifies the R_end position during the single full forward."""
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        capture["pre"] = hidden[:, r_end_pos, :].clone().cpu().float()
        if apply_fn is not None:
            new = hidden.clone()
            new[:, r_end_pos, :] = apply_fn(hidden[:, r_end_pos, :])
            capture["post"] = new[:, r_end_pos, :].clone().cpu().float()
            if isinstance(output, tuple):
                return (new,) + output[1:]
            return new
        else:
            capture["post"] = capture["pre"].clone()
            return None  # no modification
    return hook


def run_intervention(question: str, reference: str, candidate: str,
                     apply_fn=None, capture=None):
    """Monolithic full forward with an optional L18/R_end hook.

    apply_fn: callable taking (h_bf16 at R_end, shape (1, hidden)) returning
              modified (1, hidden) or None. If None, hook reads but returns None
              (zero-perturbation passive mode).
    Returns dict with l_A/l_B/d_raw/pred + captured pre/post vectors.
    """
    ids, r_end = build_prompt_ids(question, reference, candidate)
    pids = torch.tensor([ids], device="cuda")
    cap = {} if capture is None else capture
    hook = None
    if apply_fn is not None or capture is not None:
        hook = get_model().model.layers[L_BLOCK_INDEX].register_forward_hook(
            _hook_factory(r_end, apply_fn, cap))
    try:
        with torch.inference_mode():
            logits = get_model()(pids).logits
    finally:
        if hook is not None:
            hook.remove()
    ll = logits[0, pids.shape[1] - 1, :]
    l_A = ll[ACCEPT_ID].item()
    l_B = ll[REJECT_ID].item()
    d_raw = l_A - l_B
    pred = "A" if d_raw > 0 else ("B" if d_raw < 0 else "TIE")
    return {"l_A": l_A, "l_B": l_B, "d_raw": d_raw, "predicted_label": pred,
            "seq_len": int(pids.shape[1]), "r_end_pos": r_end, "capture": cap}


def load_swap_pairs(split: str):
    """Load four-cell pairs for a split from D0 preliminary_swap_pairs.jsonl."""
    pairs = [json.loads(l) for l in open(D0 / "preliminary_swap_pairs.jsonl", encoding="utf-8") if l.strip()]
    sub = [p for p in pairs if p["split"] == split]
    if len(sub) != {"train": 587, "dev": 195, "final_reserve": 197}[split]:
        raise ValueError(f"unexpected {split} pair count: {len(sub)}")
    return sub


def four_cells(pair):
    return [
        ("OO", pair["r_o"], pair["c_o"], "A"),
        ("OS", pair["r_o"], pair["c_s"], "B"),
        ("SO", pair["r_s"], pair["c_o"], "B"),
        ("SS", pair["r_s"], pair["c_s"], "A"),
    ]
