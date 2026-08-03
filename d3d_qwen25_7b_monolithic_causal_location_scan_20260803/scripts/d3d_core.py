#!/usr/bin/env python3
"""D3-D core: monolithic forward + three token-position localization
(R_end / C_end / D_pos) + arbitrary-layer arbitrary-position intervention.

固定规定：
- 全程完整单体前向，无 prefix KV cache / segmented execution。
- R_end: Reference Answer 正文最后一个非空白 token。
- C_end: Candidate Answer 正文 <answer> 的最后一个非空白 token（不含模板句号）。
- D_pos: prompt_len - 1（teacher-forced continuation 位置）。
- 干预施加于 model.model.layers[L-1] 输出、指定 token 位置。
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
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]
T0 = "The answer is <answer>."

# 候选层（hidden_states 索引）与干预层（layer block 索引）的映射
CAND_LAYERS = [14, 18, 22, 26]
POSITIONS = ["R_end", "C_end", "D_pos"]

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


def render_prompt(question, reference, candidate):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(question=question, reference=reference, candidate=candidate)},
    ]
    return get_tok().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_positions(question, reference, candidate):
    """返回 (ids, R_end, C_end, D_pos)，全部为 token 索引。"""
    tok = get_tok()
    rendered = render_prompt(question, reference, candidate)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    # R_end
    ref_marker = "Candidate Answer: "
    # reference 位于 "Reference Answer: " 之后
    r0 = rendered.find("Reference Answer: ")
    if r0 < 0:
        raise ValueError("Reference Answer marker not found")
    r_start = r0 + len("Reference Answer: ")
    r_end_char = r_start + len(reference)
    r_end = next(ti for ti, (s, e) in enumerate(offsets) if s <= r_end_char - 1 < e)
    # C_end: candidate 正文 <answer> 的最后一个非空白 token，不含模板句号
    # candidate 字符串 = T0.replace("<answer>", cand) = "The answer is {cand}."
    c0 = rendered.find("Candidate Answer: ")
    if c0 < 0:
        raise ValueError("Candidate Answer marker not found")
    c_start = c0 + len("Candidate Answer: ")
    # candidate 中 answer 正文
    cand_str = candidate
    # answer 正文在 candidate 字符串中的起止（candidate = "The answer is <answer>.")
    inner = "The answer is "
    assert cand_str.startswith(inner) and cand_str.endswith("."), f"candidate format unexpected: {cand_str[:50]!r}"
    ans_start = c_start + len(inner)
    ans_end = ans_start + (len(cand_str) - len(inner) - 1)  # 去掉尾部句号
    c_end = next(ti for ti, (s, e) in enumerate(offsets) if s <= ans_end - 1 < e)
    # D_pos
    d_pos = len(ids) - 1
    return ids, r_end, c_end, d_pos


def extract_all_positions(question, reference, candidate, layers):
    """一次完整单体前向，提取指定 layers 的 R_end/C_end/D_pos 三位置 hidden。
    返回 {f"L{li}/{pos}": (n, 3584) float32 numpy}，以及 seq_len。"""
    ids, r_end, c_end, d_pos = build_positions(question, reference, candidate)
    pids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = get_model()(pids, output_hidden_states=True)
    positions = {"R_end": r_end, "C_end": c_end, "D_pos": d_pos}
    feats = {}
    for li in layers:
        hs = out.hidden_states[li][0]  # (seq, 3584)
        for pos, pi in positions.items():
            feats[f"L{li}/{pos}"] = hs[pi].cpu().float().numpy()
    return feats, len(ids)


def extract_hidden(question, reference, candidate, layer_idx, pos):
    """完整单体前向，提取 hidden_states[layer_idx][pos]（float32 numpy）。"""
    ids, r_end, c_end, d_pos = build_positions(question, reference, candidate)
    pids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = get_model()(pids, output_hidden_states=True)
    p = {"R_end": r_end, "C_end": c_end, "D_pos": d_pos}[pos]
    h = out.hidden_states[layer_idx][0, p].cpu().float().numpy()
    return h, len(ids)


def _make_hook(token_pos, apply_fn, capture):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if apply_fn is not None:
            new = hidden.clone()
            new[:, token_pos, :] = apply_fn(hidden[:, token_pos, :])
            if isinstance(output, tuple):
                return (new,) + output[1:]
            return new
        return None
    return hook


def run_intervention(question, reference, candidate, layer_idx, pos, apply_fn=None, capture=None):
    """完整单体前向，在 layers[layer_idx-1] 输出的 token pos 处可选 patch。"""
    ids, r_end, c_end, d_pos = build_positions(question, reference, candidate)
    pids = torch.tensor([ids], device="cuda")
    token_pos = {"R_end": r_end, "C_end": c_end, "D_pos": d_pos}[pos]
    cap = {} if capture is None else capture
    hook = None
    if apply_fn is not None or capture is not None:
        hook = get_model().model.layers[layer_idx - 1].register_forward_hook(
            _make_hook(token_pos, apply_fn, cap))
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
            "seq_len": int(pids.shape[1]), "r_end_pos": r_end, "c_end_pos": c_end,
            "d_pos": d_pos, "token_pos": token_pos, "capture": cap}


def score_monolithic(question, reference, candidate):
    """无 hook 的完整单体前向，返回 A/B 读出。"""
    ids, r_end, c_end, d_pos = build_positions(question, reference, candidate)
    pids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        logits = get_model()(pids).logits
    ll = logits[0, pids.shape[1] - 1, :]
    l_A = ll[ACCEPT_ID].item()
    l_B = ll[REJECT_ID].item()
    d_raw = l_A - l_B
    pred = "A" if d_raw > 0 else ("B" if d_raw < 0 else "TIE")
    return {"l_A": l_A, "l_B": l_B, "d_raw": d_raw, "predicted_label": pred,
            "seq_len": int(pids.shape[1]), "r_end_pos": r_end, "c_end_pos": c_end, "d_pos": d_pos}


def load_swap_pairs(split):
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
