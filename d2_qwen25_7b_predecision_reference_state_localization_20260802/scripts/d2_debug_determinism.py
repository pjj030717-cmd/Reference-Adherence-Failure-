#!/usr/bin/env python3
"""Determine: is forward deterministic? Where does T0 vs T1 R_end diff occur?"""
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
print("group:", row["source_group_id"][:8], "ref:", repr(ref))


def forward(template):
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
    prompt_ids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = model(prompt_ids, output_hidden_states=True)
    hs = out.hidden_states
    h = np.stack([hs[l][0, r_tok, :].cpu().float().numpy() for l in range(1, 29)])
    return h, r_tok, len(ids)


h0a, r0a, l0a = forward(T0)
h0b, r0b, l0b = forward(T0)
h1, r1, l1 = forward(T1)
print(f"T0 run1: r_tok={r0a} len={l0a}; T0 run2: r_tok={r0b} len={l0b}")
print(f"T0-vs-T0 (determinism) max diff: {np.abs(h0a-h0b).max():.8f}")
print(f"T1-vs-T0 max diff: {np.abs(h1-h0a).max():.8f}")
for l in range(28):
    d = np.abs(h1[l] - h0a[l]).max()
    if d > 1e-3:
        print(f"  layer {l+1}: max diff {d:.4f}")
print("sequence len T0:", l0a, "T1:", l1)
