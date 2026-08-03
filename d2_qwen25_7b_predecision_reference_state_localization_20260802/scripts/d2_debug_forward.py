#!/usr/bin/env python3
"""Debug: run actual forwards on one group for T0/T1/T2, compare R_end hidden states per layer."""
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
r = dev_ss[0]
q, ref = r["question"], r["reference"]

states = {}
for name, t in (("T0", T0), ("T1", T1), ("T2", T2)):
    cand = t.replace("<answer>", ref)
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
    states[name] = {"r_tok": r_tok, "hs": [hs[l][0, r_tok, :].cpu().float().numpy() for l in range(1, 29)]}
    print(f"{name}: r_tok={r_tok}")

for name in ("T1", "T2"):
    for l in range(28):
        d = np.abs(states[name]["hs"][l] - states["T0"]["hs"][l]).max()
        if d > 1e-5:
            print(f"{name} layer {l+1} max abs diff vs T0: {d:.6f}")

# Also compare collected T0 saved npz
g = r["source_group_id"]
z = np.load(D2 / "hidden_states" / f"dev_{g}.npz")["SS_h_r"]
d_saved = np.abs(z.astype(np.float32) - np.stack(states["T0"]["hs"])).max()
print("collected-vs-recomputed T0 max diff:", d_saved)
