#!/usr/bin/env python3
"""Truncation experiment: does T1 R_end differ because of the suffix after it?"""
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

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()

dev_ss = [r for r in json.loads((D2 / "scripts" / "_dev_rows.json").read_text(encoding="utf-8"))
          if r["cell"] == "SS"]
row = [r for r in dev_ss if r["source_group_id"].startswith("fe8d6ea3")][0]
q, ref = row["question"], row["reference"]


def make_inputs(template, truncate=None):
    cand = template.replace("<answer>", ref)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(question=q, reference=ref, candidate=cand)}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    ref_marker = "Reference Answer: "
    i0 = rendered.find(ref_marker)
    ref_end = i0 + len(ref_marker) + len(ref)
    r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
    if truncate is not None:
        ids = ids[:r_tok + truncate]
    return ids, r_tok


def get_h(ids, r_tok):
    prompt_ids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = model(prompt_ids, output_hidden_states=True)
    hs = out.hidden_states
    return np.stack([hs[l][0, r_tok, :].cpu().float().numpy() for l in range(1, 29)])


ids0, r0 = make_inputs(T0)
h0 = get_h(ids0, r0)
print("T0 len:", len(ids0), "r_tok:", r0)

ids1, r1 = make_inputs(T1)
h1_full = get_h(ids1, r1)
print("T1 len:", len(ids1), "r_tok:", r1)
print("T1 full vs T0:", np.abs(h1_full - h0).max())

# truncate T1 to r_tok+1 (prefix only)
ids1t, _ = make_inputs(T1)
ids1t = ids1t[:r1 + 1]
h1_t = get_h(ids1t, r1)
print("T1 truncated (prefix only) vs T0:", np.abs(h1_t - h0).max())

# truncate T1 to r_tok+2 (one token after R_end)
ids1b = ids1[:r1 + 2]
h1_b = get_h(ids1b, r1)
print("T1 prefix+1token vs T0:", np.abs(h1_b - h0).max())
