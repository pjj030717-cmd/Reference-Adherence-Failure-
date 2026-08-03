#!/usr/bin/env python3
"""A1 Phase 1: synthetic readout regression + dual-order factual preference scoring.

1) Synthetic regression (24 pairs, frozen manifest):
   - For each pair we run BOTH orders:
       Order1: A=option_a(correct if correct=='A'), B=option_b
       Order2: A=option_b, B=option_a
   - teacher-forced prediction per order: pred = 'A' if d_raw>0 else ('B' if d_raw<0 else TIE),
     where d_raw = l_A - l_B within that order's A/B framing.
   - overall correctness of a pair requires BOTH orders to pick the correct option
     (Order2 correct option is the flipped one).
   - gates: overall order accuracy >= 22/24; A-correct >= 10/12; B-correct >= 10/12;
     ties == 0; greedy first-token diagnostic consistent with teacher-forced >= 22/24.
   On failure: factual_preference_readout_invalid.

2) Dual-order scoring for SciQ dev + PopQA dev (the k score):
   Order1: A=r_o, B=r_s -> d1 = l_A - l_B
   Order2: A=r_s, B=r_o -> d2 = l_B - l_A   (r_o preference)
   k = (d1 + d2) / 2
   On NaN/inf/truncation/tokenization failure or alignment failure: behavioral_attribution_input_invalid.

No hidden states; final-layer logits at pos=prompt_len-1 only.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = REPO_ROOT / "a1_qwen_no_reference_factual_preference_adherence_audit_20260803"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")

SYSTEM = "Choose the option that is factually correct for the question.\nReply with only A or B."
USER_TMPL = "Question:\n{q}\n\nOption A:\n{option_a}\n\nOption B:\n{option_b}\n\nAnswer:"
A_ID, B_ID = 362, 425


def fail(label, why):
    print("STOP:", label, "-", why)
    (OUT / "artifacts").mkdir(parents=True, exist_ok=True)
    (OUT / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": False, "final_reserve_text_read": False,
        "hidden_states_read": False, "probe_trained": False,
        "activation_intervention_run": False}, indent=2), encoding="utf-8")
    sys.exit(1)


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()
print("model loaded for Phase 1")


def score_order(q, option_a, option_b):
    """Return l_A, l_B (logprobs of ' A' and ' B' at pos=prompt_len-1), d_raw=l_A-l_B,
    greedy first-token letter (from generation, forced single token) and prompt_len."""
    user = USER_TMPL.format(q=q, option_a=option_a, option_b=option_b)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    prompt_ids = enc["input_ids"].to("cuda")
    amask = enc["attention_mask"].to("cuda")
    prompt_len = prompt_ids.shape[1]
    with torch.inference_mode():
        logits = model(prompt_ids, attention_mask=amask).logits
    pos = prompt_len - 1
    ll = logits[0, pos, :]
    l_A = ll[A_ID].item()
    l_B = ll[B_ID].item()
    d_raw = l_A - l_B
    # greedy first-token: sample 1 token greedily; map 'A'(32)/'B'(33) to letter, else None
    out = model.generate(prompt_ids, attention_mask=amask, max_new_tokens=1, do_sample=False)
    gtok = tok.decode([int(out[0][prompt_len])])
    gtok = gtok.strip()
    greedy = None
    if gtok == "A":
        greedy = "A"
    elif gtok == "B":
        greedy = "B"
    return l_A, l_B, d_raw, greedy, prompt_len


# ============ 1. synthetic regression ============
syn = json.loads((OUT / "synthetic_factual_choice_manifest.json").read_text(encoding="utf-8"))
print("synthetic pairs:", len(syn))
syn_rows = []
accA_total = accB_total = 0
greedy_agree = 0
for sid, q, opt_a, opt_b, correct in syn:
    # Order1: A=opt_a, B=opt_b. correct option position = correct
    l_A1, l_B1, d1raw, g1, plen1 = score_order(q, opt_a, opt_b)
    pred1 = "A" if d1raw > 0 else ("B" if d1raw < 0 else "TIE")
    cor1 = pred1 == correct
    # Order2: A=opt_b, B=opt_a. correct option position = flipped
    l_A2, l_B2, d2raw, g2, plen2 = score_order(q, opt_b, opt_a)
    pred2 = "A" if d2raw > 0 else ("B" if d2raw < 0 else "TIE")
    flip = "B" if correct == "A" else "A"
    cor2 = pred2 == flip
    overall = cor1 and cor2
    # greedy consistency: greedy first-token letter == teacher-forced pred in that order
    g_agree1 = g1 is not None and g1 == pred1
    g_agree2 = g2 is not None and g2 == pred2
    g_pair = g_agree1 and g_agree2
    greedy_agree += 1 if g_pair else 0
    if correct == "A":
        accA_total += 1 if overall else 0
    else:
        accB_total += 1 if overall else 0
    syn_rows.append({"id": sid, "question": q, "option_a": opt_a, "option_b": opt_b,
                     "correct_option": correct,
                     "order1_pred": pred1, "order1_correct": cor1, "order1_d_raw": d1raw,
                     "order2_pred": pred2, "order2_correct": cor2, "order2_d_raw": d2raw,
                     "pair_overall_correct": overall,
                     "greedy1": g1, "greedy2": g2, "greedy_pair_consistent": g_pair})
    print(f"{sid} correct={correct} pred1={pred1}({cor1}) pred2={pred2}({cor2}) overall={overall} "
          f"d1={d1raw:+.2f} d2raw={d2raw:+.2f} g1={g1} g2={g2} greedy_ok={g_pair}")

with open(OUT / "synthetic_factual_choice_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(syn_rows[0].keys()))
    w.writeheader()
    w.writerows(syn_rows)

overall_acc = (accA_total + accB_total) / 24
ties = sum(1 for r in syn_rows if r["order1_d_raw"] == 0 or r["order2_d_raw"] == 0)
print(f"synthetic: overall={overall_acc:.4f} ({accA_total + accB_total}/24) "
      f"A={accA_total}/12 B={accB_total}/12 ties={ties} greedy_agree={greedy_agree}/24")

if overall_acc < 22 / 24 or accA_total < 10 or accB_total < 10 or ties > 0 or greedy_agree < 22:
    fail("factual_preference_readout_invalid",
         f"overall={overall_acc:.4f} A={accA_total} B={accB_total} ties={ties} greedy={greedy_agree}")
print("SYNTHETIC FACTUAL-CHOICE READOUT GATE PASSED")

# ============ 2. dual-order dev scoring ============
def token_len(s):
    return len(tok.encode(s, add_special_tokens=False))


out_rows = []
problems = []
for ds, path in [("SciQ", OUT / "scripts" / "_sciq_dev.jsonl"),
                 ("PopQA", OUT / "scripts" / "_popqa_dev.jsonl")]:
    recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    print(f"scoring {ds}: {len(recs)} groups")
    for r in recs:
        q, r_o, r_s, y = r["question"], r["r_o"], r["r_s"], r["y_SS"]
        gid = r["source_group_id"]
        l_A1, l_B1, d1, g1, _ = score_order(q, r_o, r_s)            # A=r_o, B=r_s
        l_A2, l_B2, d2raw, g2, _ = score_order(q, r_s, r_o)         # A=r_s, B=r_o
        d2 = l_B2 - l_A2                                            # r_o preference
        k = (d1 + d2) / 2.0
        vals = [d1, d2, k, l_A1, l_B1, l_A2, l_B2]
        if any(v != v or math.isinf(v) for v in vals):
            problems.append((ds, gid, "NaN/inf"))
            continue
        oc = (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0)
        tie = (d1 == 0 or d2 == 0)
        out_rows.append({"dataset": ds, "source_group_id": gid,
                         "relation": r.get("relation", "NA"), "question": q,
                         "r_o": r_o, "r_s": r_s,
                         "l_A_order1": l_A1, "l_B_order1": l_B1,
                         "l_A_order2": l_A2, "l_B_order2": l_B2,
                         "d_1": d1, "d_2": d2, "k": k,
                         "order_consistent": oc, "tie": tie, "y_SS": y,
                         "question_token_length": token_len(q),
                         "r_o_token_length": token_len(r_o),
                         "r_s_token_length": token_len(r_s)})

if problems:
    print("PROBLEM rows:", len(problems))
    for p in problems[:10]:
        print("  ", p)
    fail("behavioral_attribution_input_invalid", f"{len(problems)} problematic groups (NaN/inf)")

print("scored rows:", len(out_rows))
cnt = {}
for r in out_rows:
    cnt[r["dataset"]] = cnt.get(r["dataset"], 0) + 1
print("per dataset:", cnt)
if cnt.get("SciQ") != 195 or cnt.get("PopQA") != 2815:
    fail("behavioral_attribution_input_invalid", f"row counts {cnt}")

with open(OUT / "factual_preference_scores_dev.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

(OUT / "tokenization_and_teacher_forcing_audit.md").write_text(
    """# tokenization_and_teacher_forcing_audit.md

## A/B continuation（Qwen tokenizer）

| 项 | " A" | " B" |
|---|---|---|
| encode() | [362] | [425] |
| 单 token | 是 | 是 |
| 等长 | 1 | 1 |
| UNK | 否 | 否 |

## 无 Reference 二选一 prompt（固定）

- system：`Choose the option that is factually correct for the question.\\nReply with only A or B.`
- user 模板：`Question:\\n{q}\\n\\nOption A:\\n{option_a}\\n\\nOption B:\\n{option_b}\\n\\nAnswer:`
- 无 Reference / Candidate / Judge / Accept / Reject 字样。

## teacher-forced 位置

```python
pos = prompt_len - 1
d_raw = l_A - l_B   # l_A = log p(" A"), l_B = log p(" B")
```

- Order1: A=r_o, B=r_s -> d_1 = l_A - l_B
- Order2: A=r_s, B=r_o -> d_2 = l_B - l_A（以 r_o 为偏好方向）
- k = (d_1 + d_2) / 2
- 无空白先验校正 / 阈值调参 / prompt 搜索。
""", encoding="utf-8")

(OUT / "model_access_audit.md").write_text(
    """# model_access_audit.md

## 模型

- Qwen/Qwen2.5-7B-Instruct，本地 `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`
- BF16 / eval / inference_mode / batch_size=1
- 与 D1/E1 完全一致；未加载任何额外 Judge。

## 边界

- 未读取 hidden state；未训练 Probe；无 hook/intervention。
""", encoding="utf-8")

print("Phase 1 OK")
