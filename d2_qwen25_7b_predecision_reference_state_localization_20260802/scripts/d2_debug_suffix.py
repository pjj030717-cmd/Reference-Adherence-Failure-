#!/usr/bin/env python3
"""Decisive test: same prefix, different suffix lengths -> does R_end hidden change?"""
import json
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
T0 = "The answer is <answer>."
T1 = "For this question, the answer is <answer>."
T2 = "The response is <answer>."

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()

dev_ss = [r for r in json.loads((D2 / "scripts" / "_dev_rows.json").read_text(encoding="utf-8"))
          if r["cell"] == "SS"]
row = [r for r in dev_ss if r["source_group_id"].startswith("fe8d6ea3")][0]
q, ref = row["question"], row["reference"]


def build(question, template):
    cand = template.replace("<answer>", ref)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(question=question, reference=ref, candidate=cand)}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    ref_marker = "Reference Answer: "
    i0 = rendered.find(ref_marker)
    ref_end = i0 + len(ref_marker) + len(ref)
    r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
    return ids, r_tok


def get_h(ids, r_tok):
    prompt_ids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = model(prompt_ids, output_hidden_states=True)
    hs = out.hidden_states
    return np.stack([hs[l][0, r_tok, :].cpu().float().numpy() for l in range(1, 29)])


# same prefix (same rendered), different suffix: append junk tokens of varying length
ids0, r0 = build(q, T0)
h0 = get_h(ids0, r0)
print(f"T0 base len={len(ids0)} r_tok={r0}")

for extra in (1, 4, 10, 100):
    idsx = list(ids0) + list(tok.encode(" extra token") * extra)  # arbitrary suffix tokens
    hx = get_h(idsx, r0)
    print(f"  +{extra} junk suffix tokens (len={len(idsx)}): R_end max diff vs base = {np.abs(hx-h0).max():.6f}")

# same total length but different suffix content (same len as T1 = 129)
ids1, r1 = build(q, T1)
h1 = get_h(ids1, r1)
print(f"T1 len={len(ids1)} R_end diff vs T0: {np.abs(h1-h0).max():.6f}")

# append junk to reach same 129 length as T1
pad_len = len(ids1) - len(ids0)
ids_pad = list(ids0) + list(tok.encode(" abcdefghijklmno"))[:pad_len]
hpad = get_h(ids_pad, r0)
print(f"T0+{pad_len} junk tokens to match T1 len: R_end diff = {np.abs(hpad-h0).max():.6f}")
