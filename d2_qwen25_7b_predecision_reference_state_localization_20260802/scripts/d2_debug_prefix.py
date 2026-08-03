#!/usr/bin/env python3
"""Check whether prefix token ids up to R_end differ between T0 and T1."""
import json
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]
from transformers import AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
T0 = "The answer is <answer>."
T1 = "For this question, the answer is <answer>."

tok = AutoTokenizer.from_pretrained(MODEL)
dev_ss = [r for r in json.loads((D2 / "scripts" / "_dev_rows.json").read_text(encoding="utf-8"))
          if r["cell"] == "SS"]

bad = {"fe8d6ea33b2967974ed5d5bb87deaec20c56d3847aeb51fca31b56797035a3a8",
       "f2a0372b48ecb146721a5a97015730fd57d998767bee00a9570618d5b4a15334"}


def get_info(row, template):
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
    return ids, r_tok, ref_end, rendered


for row in dev_ss:
    if row["source_group_id"] not in bad:
        continue
    ids0, r0, re0, rend0 = get_info(row, T0)
    ids1, r1, re1, rend1 = get_info(row, T1)
    print(f"\ngroup={row['source_group_id'][:8]} ref={row['reference']!r}")
    print(f"  len0={len(ids0)} len1={len(ids1)} r0={r0} r1={r1}")
    print(f"  prefix ids equal: {ids0[:r0+1] == ids1[:r0+1]}")
    # find first diff
    for i, (a, b) in enumerate(zip(ids0[:r0 + 1], ids1[:r0 + 1])):
        if a != b:
            print(f"  first prefix diff at idx {i}: T0={tok.decode([a])!r} T1={tok.decode([b])!r}")
            # print context of rendered
            print("  T0 rendered ...", rend0[:120])
            print("  T1 rendered ...", rend1[:120])
            break
    else:
        print("  no prefix diff found?!")
