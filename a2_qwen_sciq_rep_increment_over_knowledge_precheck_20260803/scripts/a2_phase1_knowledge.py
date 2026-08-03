#!/usr/bin/env python3
"""A2 Phase 1: frozen B_knowledge scoring.

1. A1 synthetic 24-pair factual-choice readout regression reproduction
   (manifest from A1; must pass overall=24/24, A=12/12, B=12/12, ties=0,
   greedy-consistent>=22). Failure -> knowledge_score_readout_invalid.
2. Dual-order teacher-forced k scoring on SciQ train (587) and dev (195):
   Order1 A=r_o,B=r_s -> d_1 = l_A - l_B
   Order2 A=r_s,B=r_o -> d_2 = l_B - l_A
   k = (d_1 + d_2) / 2
   A1-fixed system/user template, continuation tokens 362/425, pos=prompt_len-1.

Writes _k_train.json / _k_dev.json and knowledge_score_reproduction_audit.md.
No hidden states, no PopQA, no final-reserve.
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

OUT = REPO_ROOT / "a2_qwen_sciq_rep_increment_over_knowledge_precheck_20260803"
A1 = REPO_ROOT / "a1_qwen_no_reference_factual_preference_adherence_audit_20260803"
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
        "hidden_states_newly_extracted": False, "popqa_read": False,
        "prompt_searched": False, "activation_intervention_run": False}, indent=2), encoding="utf-8")
    sys.exit(1)


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()
print("model loaded for Phase 1")


def score_order(q, option_a, option_b):
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
    out = model.generate(prompt_ids, attention_mask=amask, max_new_tokens=1, do_sample=False)
    gtok = tok.decode([int(out[0][prompt_len])]).strip()
    greedy = gtok if gtok in ("A", "B") else None
    return l_A, l_B, d_raw, greedy, prompt_len


# ---------------------------------------------------------------------------
# 1. synthetic readout regression reproduction (A1 manifest)
# ---------------------------------------------------------------------------
syn = json.loads((A1 / "synthetic_factual_choice_manifest.json").read_text(encoding="utf-8"))
assert len(syn) == 24
syn_rows = []
acc_a = acc_b = greedy_agree = 0
for sid, q, opt_a, opt_b, correct in syn:
    l_A1, l_B1, d1, g1, _ = score_order(q, opt_a, opt_b)
    l_A2, l_B2, d2raw, g2, _ = score_order(q, opt_b, opt_a)
    pred1 = "A" if d1 > 0 else ("B" if d1 < 0 else "TIE")
    pred2 = "A" if d2raw > 0 else ("B" if d2raw < 0 else "TIE")
    flip = "B" if correct == "A" else "A"
    cor1, cor2 = pred1 == correct, pred2 == flip
    overall = cor1 and cor2
    if correct == "A":
        acc_a += 1 if overall else 0
    else:
        acc_b += 1 if overall else 0
    g_ok = (g1 is not None and g1 == pred1) and (g2 is not None and g2 == pred2)
    greedy_agree += 1 if g_ok else 0
    syn_rows.append({"id": sid, "pred1": pred1, "pred2": pred2, "cor1": cor1, "cor2": cor2,
                     "overall": overall, "d1": d1, "d2raw": d2raw, "g1": g1, "g2": g2})
    print(f"{sid} correct={correct} pred1={pred1}({cor1}) pred2={pred2}({cor2}) overall={overall} g={g_ok}")

ties = sum(1 for r in syn_rows if r["d1"] == 0 or r["d2raw"] == 0)
total = acc_a + acc_b
print(f"synthetic reproduction: overall={total}/24 A={acc_a}/12 B={acc_b}/12 ties={ties} greedy={greedy_agree}/24")
if total < 24 or acc_a < 12 or acc_b < 12 or ties > 0 or greedy_agree < 22:
    fail("knowledge_score_readout_invalid",
         f"overall={total}/24 A={acc_a}/12 B={acc_b}/12 ties={ties} greedy={greedy_agree}/24")

# ---------------------------------------------------------------------------
# 2. dual-order k scoring on train + dev
# ---------------------------------------------------------------------------
def token_len(s):
    return len(tok.encode(s, add_special_tokens=False))


def score_k(meta, split):
    out = []
    for r in meta:
        q, r_o, r_s = r["question"], r["r_o"], r["r_s"]
        l_A1, l_B1, d1, _, _ = score_order(q, r_o, r_s)
        l_A2, l_B2, d2raw, _, _ = score_order(q, r_s, r_o)
        d2 = l_B2 - l_A2
        k = (d1 + d2) / 2.0
        vals = [d1, d2, k, l_A1, l_B1, l_A2, l_B2]
        if any(v != v or math.isinf(v) for v in vals):
            fail("knowledge_score_readout_invalid", f"{split} group {r['source_group_id']} NaN/inf")
        out.append({**r, "d_1": d1, "d_2": d2, "k": k,
                    "l_A1": l_A1, "l_B1": l_B1, "l_A2": l_A2, "l_B2": l_B2,
                    "question_token_length": token_len(q),
                    "r_o_token_length": token_len(r_o),
                    "r_s_token_length": token_len(r_s)})
    return out


meta_tr = json.loads((OUT / "scripts" / "_meta_train.json").read_text(encoding="utf-8"))
meta_de = json.loads((OUT / "scripts" / "_meta_dev.json").read_text(encoding="utf-8"))
k_tr = score_k(meta_tr, "train")
k_de = score_k(meta_de, "dev")
print(f"scored k_train={len(k_tr)} k_dev={len(k_de)}")

(OUT / "scripts" / "_k_train.json").write_text(json.dumps(k_tr, ensure_ascii=False), encoding="utf-8")
(OUT / "scripts" / "_k_dev.json").write_text(json.dumps(k_de, ensure_ascii=False), encoding="utf-8")

# ---------------------------------------------------------------------------
# knowledge_score_reproduction_audit.md
# ---------------------------------------------------------------------------
(OUT / "knowledge_score_reproduction_audit.md").write_text(
    f"""# knowledge_score_reproduction_audit.md

## A1 冻结规范（完全继承，未改写）

- system：`Choose the option that is factually correct for the question.\\nReply with only A or B.`
- user 模板：`Question:\\n{{q}}\\n\\nOption A:\\n{{option_a}}\\n\\nOption B:\\n{{option_b}}\\n\\nAnswer:`
- 选项直接填 `r_o` / `r_s` 字符串，无 Candidate/Reference/Judge 句式。
- continuation：`" A"`=362、`" B"`=425（单 token、等长、非 UNK）。
- teacher-forcing 位置：`pos = prompt_len - 1`。
- Order1: A=r_o,B=r_s → d_1 = l_A − l_B；Order2: A=r_s,B=r_o → d_2 = l_B − l_A；k = (d_1+d_2)/2。
- 无空白先验校正、无阈值调参、无 prompt 搜索。

## 合成 24 对回归复现（A1 manifest 冻结原样）

| 检查 | 要求 | 本次结果 |
|---|---|---|
| overall | 24/24 | {total}/24 |
| A-correct | 12/12 | {acc_a}/12 |
| B-correct | 12/12 | {acc_b}/12 |
| ties | 0 | {ties} |
| greedy 一致 | >=22/24 | {greedy_agree}/24 |

## k 评分范围

- SciQ train：587 groups（`_k_train.json`）
- SciQ dev：195 groups（`_k_dev.json`）
- 无 NaN/inf；未接触 PopQA、final-reserve。
""", encoding="utf-8")

# write synthetic audit csv for traceability
with open(OUT / "scripts" / "_synthetic_reproduction.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(syn_rows[0].keys()))
    w.writeheader()
    w.writerows(syn_rows)

print("Phase 1 OK")
