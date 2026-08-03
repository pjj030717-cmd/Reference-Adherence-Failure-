#!/usr/bin/env python3
"""Check whether logits at R_end also change with total sequence length."""
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

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()

dev_ss = [r for r in json.loads((D2 / "scripts" / "_dev_rows.json").read_text(encoding="utf-8"))
          if r["cell"] == "SS"]
row = [r for r in dev_ss if r["source_group_id"].startswith("fe8d6ea3")][0]
q, ref = row["question"], row["reference"]
cand = T0.replace("<answer>", ref)
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
print("base len:", len(ids), "r_tok:", r_tok)

ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]


def get_logits(token_ids):
    prompt_ids = torch.tensor([token_ids], device="cuda")
    with torch.inference_mode():
        out = model(prompt_ids)
    logits = out.logits[0, r_tok]
    return logits[ACCEPT_ID].item(), logits[REJECT_ID].item()


la0, lb0 = get_logits(ids)
print(f"base: l_A={la0:.6f} l_B={lb0:.6f}")
for n in (1, 4, 20):
    idsx = list(ids) + list(tok.encode(" extra token") * n)
    la, lb = get_logits(idsx)
    print(f"+{2*n} junk tokens (len={len(idsx)}): l_A={la:.6f} (delta {la-la0:+.6f}) l_B={lb:.6f} (delta {lb-lb0:+.6f})")
