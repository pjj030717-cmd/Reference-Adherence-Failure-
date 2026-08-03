#!/usr/bin/env python3
"""Analyze the 27 groups whose T1 R_end differs from T0: label pattern + features."""
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
    ref_end = i0 + len(ref_marker) + len(ref)
    r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
    prompt_ids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = model(prompt_ids, output_hidden_states=True)
    hs = out.hidden_states
    h = np.stack([hs[l][0, r_tok, :].cpu().float().numpy() for l in range(1, 29)])
    return h, r_tok, len(ids)


rows = []
for row in dev_ss:
    h0, r0, l0 = get_r_end(row, T0)
    h1, r1, l1 = get_r_end(row, T1)
    d = float(np.abs(h1 - h0).max())
    y = 1 if row["predicted_label"] == "B" else 0
    ref = row["reference"]
    rows.append({"gid": row["source_group_id"], "diff": d, "y": y, "ref": ref,
                 "ref_words": len(ref.split()), "ref_len": len(ref),
                 "l0": l0, "l1": l1, "r_tok": r0})

affected = [r for r in rows if r["diff"] > 1e-4]
print(f"affected groups: {len(affected)} / {len(rows)}")
print(f"  affected y=1 (SS error): {sum(1 for r in affected if r['y']==1)} / {sum(1 for r in rows if r['y']==1)} total y=1")
print(f"  affected y=0 (SS correct): {sum(1 for r in affected if r['y']==0)} / {sum(1 for r in rows if r['y']==0)} total y=0")
print(f"  affected ref multiword: {sum(1 for r in affected if r['ref_words']>1)} / {sum(1 for r in rows if r['ref_words']>1)} total multiword")
print(f"  affected ref_len mean: {np.mean([r['ref_len'] for r in affected]):.1f} vs all {np.mean([r['ref_len'] for r in rows]):.1f}")
print(f"  affected l0 mean: {np.mean([r['l0'] for r in affected]):.1f} vs all {np.mean([r['l0'] for r in rows]):.1f}")
print(f"  affected r_tok range: {min(r['r_tok'] for r in affected)}..{max(r['r_tok'] for r in affected)}")
print("  affected refs:", [r['ref'] for r in affected][:20])

# check whether affected diff is concentrated at high layers
worst = max(affected, key=lambda r: r['diff'])
print("\nworst group:", worst['ref'], "diff:", worst['diff'])
