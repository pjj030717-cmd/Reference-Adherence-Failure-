#!/usr/bin/env python3
"""Diagnose per-group T1 R_end differences vs T0."""
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


def get_r_end(row, template):
    q, ref = row["question"], row["reference"]
    cand = template.replace("<answer>", ref)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(question=q, reference=ref, candidate=cand)}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    ref_marker = "Reference Answer: "
    i0 = rendered.find(ref_marker)
    ref_start = i0 + len(ref_marker)
    ref_end = ref_start + len(ref)
    r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
    prompt_ids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = model(prompt_ids, output_hidden_states=True)
        hs = out.hidden_states
    h = np.stack([hs[l][0, r_tok, :].cpu().float().numpy() for l in range(1, 29)])
    return h, r_tok, ref_end


worst = []
for i, row in enumerate(dev_ss):
    h0, r0, re0 = get_r_end(row, T0)
    h1, r1, re1 = get_r_end(row, T1)
    d = np.abs(h1 - h0).max()
    worst.append((d, i, row["source_group_id"], r0, r1, re0, re1, row["reference"]))

worst.sort(reverse=True)
print("top 5 worst T1-vs-T0 R_end diffs:")
for d, i, g, r0, r1, re0, re1, ref in worst[:5]:
    print(f"  group={g} maxdiff={d:.5f} r_tok_T0={r0} r_tok_T1={r1} ref_end_T0={re0} ref_end_T1={re1} ref={ref!r}")
print("groups with diff > 1e-4:", sum(1 for w in worst if w[0] > 1e-4))
