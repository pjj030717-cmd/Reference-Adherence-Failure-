#!/usr/bin/env python3
"""Test: eager attention — does T1 full-sequence R_end equal T0?"""
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

dev_ss = [r for r in json.loads((D2 / "scripts" / "_dev_rows.json").read_text(encoding="utf-8"))
          if r["cell"] == "SS"]
row = [r for r in dev_ss if r["source_group_id"].startswith("fe8d6ea3")][0]
q, ref = row["question"], row["reference"]


def make_inputs(template):
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
    return ids, r_tok


ids0, r0 = make_inputs(T0)

for attn in ("eager", "sdpa"):
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True,
                                                 attn_implementation=attn)
    model.eval()
    print(f"\n=== attn_implementation = {attn} ===")
    outs = {}
    for name, t in (("T0", T0), ("T1", T1), ("T2", T2)):
        ids, r_tok = make_inputs(t)
        prompt_ids = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            out = model(prompt_ids, output_hidden_states=True)
        hs = out.hidden_states
        h = np.stack([hs[l][0, r_tok, :].cpu().float().numpy() for l in range(1, 29)])
        outs[name] = h
        print(f"  {name} len={len(ids)} r_tok={r_tok}")
    print(f"  T1-vs-T0: {np.abs(outs['T1']-outs['T0']).max():.6f}")
    print(f"  T2-vs-T0: {np.abs(outs['T2']-outs['T0']).max():.6f}")
    del model
