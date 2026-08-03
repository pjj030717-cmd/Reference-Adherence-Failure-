#!/usr/bin/env python3
"""Debug: compare R_end token position/index across T0/T1/T2 for one group."""
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
T2 = "The response is <answer>."

tok = AutoTokenizer.from_pretrained(MODEL)
dev_ss = [r for r in json.loads((D2 / "scripts" / "_dev_rows.json").read_text(encoding="utf-8"))
          if r["cell"] == "SS"]

r = dev_ss[0]
q, ref = r["question"], r["reference"]
print("group:", r["source_group_id"], "ref:", repr(ref))

for name, t in (("T0", T0), ("T1", T1), ("T2", T2)):
    cand = t.replace("<answer>", ref)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(question=q, reference=ref, candidate=cand)},
    ]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    ref_marker = "Reference Answer: "
    i0 = rendered.find(ref_marker)
    ref_start = i0 + len(ref_marker)
    ref_end = ref_start + len(ref)
    r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
    # also find the LAST whitespace-trimmed char token of reference
    r_tok2 = None
    for ti in range(len(offsets) - 1, -1, -1):
        s, e = offsets[ti]
        if e <= ref_end and rendered[s:e].strip():
            r_tok2 = ti
            break
    rs, re_ = offsets[r_tok]
    print(f"\n{name}: len={len(ids)} ref_end={ref_end}")
    print(f"  r_tok={r_tok} span={offsets[r_tok]} text={rendered[rs:re_]!r}")
    print(f"  last non-ws tok={r_tok2} text={rendered[offsets[r_tok2][0]:offsets[r_tok2][1]]!r}")
    # prefix token ids before r_tok
    print(f"  prefix len to r_tok: {r_tok} tokens")
    # candidate marker position
    cand_marker = "Candidate Answer: "
    ci = rendered.find(cand_marker)
    print(f"  cand_start={ci}")
    # compare prefix ids
    if name == "T0":
        ref_prefix_ids = ids[:r_tok + 1]
    else:
        same = ids[:r_tok + 1] == ref_prefix_ids
        print(f"  prefix ids (incl r_tok) == T0: {same}, r_tok same idx: {r_tok == r_tok0}")
