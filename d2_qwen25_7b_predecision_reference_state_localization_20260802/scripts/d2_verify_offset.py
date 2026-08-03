#!/usr/bin/env python3
"""E01-D2: verify offset-mapping approach for locating R_end/C_end/D_pos.

Requirement: offset-tokenization ids == apply_chat_template(tokenize=True) ids.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

from transformers import AutoTokenizer

D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
T0 = "The answer is <answer>."

tok = AutoTokenizer.from_pretrained(MODEL)


def fail(why: str):
    print("token_span_mapping_invalid:", why)
    (D2 / "artifacts").mkdir(parents=True, exist_ok=True)
    (D2 / "artifacts" / "decision.json").write_text(
        json.dumps({"final_label": "token_span_mapping_invalid", "reason": why,
                    "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
                    "probe_trained": True, "activation_intervention_run": False,
                    "prompt_baselines_run": False, "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


def build_candidate(ans: str) -> str:
    return T0.replace("<answer>", ans)


# a probe: question/reference/candidate with distinct markers to test span finding
q = "What is the capital city of France?"
r = "Paris"
c = build_candidate("Paris")

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER_TMPL.format(question=q, reference=r, candidate=c)},
]

# 1. rendered string
rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("rendered:\n", rendered)

# 2. ids from apply_chat_template tokenize=True
ids_ct = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
print("apply_chat_template return type:", type(ids_ct))
if hasattr(ids_ct, "input_ids"):
    ids_ct = ids_ct["input_ids"]
if isinstance(ids_ct, list) and len(ids_ct) >= 1 and isinstance(ids_ct[0], list):
    ids_ct = ids_ct[0]
ids_ct = list(ids_ct)
print("ids_ct len:", len(ids_ct))

# 3. offset-tokenization
enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
ids_off = enc["input_ids"]
offsets = enc["offset_mapping"]
print("ids_ct len:", len(ids_ct), "ids_off len:", len(ids_off))
print("ids identical:", ids_ct == ids_off)

if ids_ct != ids_off:
    # show first diff
    for i, (a, b) in enumerate(zip(ids_ct, ids_off)):
        if a != b:
            print(f"first diff at {i}: ct={a}({tok.decode([a])!r}) off={b}({tok.decode([b])!r})")
            break
    fail("offset-tokenization ids differ from apply_chat_template ids")

# 4. find reference/candidate spans
ref_marker = "Reference Answer: "
cand_marker = "Candidate Answer: "
# find reference value span
i0 = rendered.find(ref_marker)
if i0 < 0:
    fail("Reference Answer marker not found in rendered prompt")
ref_start = i0 + len(ref_marker)
ref_text = r
ref_end = ref_start + len(ref_text)

i1 = rendered.find(cand_marker)
if i1 < 0:
    fail("Candidate Answer marker not found")
cand_start = i1 + len(cand_marker)
cand_text = c
cand_end = cand_start + len(cand_text)

# locate token containing char position (end-1)
def token_for_char(char_pos: int):
    """return token index whose span contains char_pos"""
    for ti, (s, e) in enumerate(offsets):
        if s <= char_pos < e:
            return ti
    return None

r_tok = token_for_char(ref_end - 1)
c_tok = token_for_char(cand_end - 1)
d_pos = len(ids_ct) - 1
print(f"ref span chars [{ref_start},{ref_end}) text={rendered[ref_start:ref_end]!r}")
print(f"cand span chars [{cand_start},{cand_end}) text={rendered[cand_start:cand_end]!r}")
print(f"R_end token idx={r_tok} text={tok.decode([ids_ct[r_tok]])!r}")
print(f"C_end token idx={c_tok} text={tok.decode([ids_ct[c_tok]])!r}")
print(f"D_pos idx={d_pos} text={tok.decode([ids_ct[d_pos]])!r}")

# verify R_end is in reference body (not field name/newline/candidate/Answer:)
rs, re_ = offsets[r_tok]
print(f"R_end span chars [{rs},{re_}) text={rendered[rs:re_]!r}")
assert rs < ref_end and re_ > ref_start, "R_end not inside reference span"
# check the char at position ref_end-1 is non-whitespace
assert not rendered[ref_end - 1].isspace(), "R_end char is whitespace!"
# C_end similar
cs, ce = offsets[c_tok]
print(f"C_end span chars [{cs},{ce}) text={rendered[cs:ce]!r}")
assert cs < cand_end and ce > cand_start
assert not rendered[cand_end - 1].isspace(), "C_end char is whitespace!"
# D_pos last token should be after Answer:
last_tok = tok.decode([ids_ct[d_pos]])
print("D_pos token text:", repr(last_tok))

print("TOKEN SPAN MAPPING APPROACH VERIFIED")
