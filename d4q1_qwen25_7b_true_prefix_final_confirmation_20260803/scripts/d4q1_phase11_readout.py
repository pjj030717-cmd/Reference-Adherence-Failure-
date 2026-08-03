#!/usr/bin/env python3
"""D4-Q1 Phase 1.1: Qwen A/B readout semantic regression on the 24 synthetic
pairs inherited from D1 (12 MATCH->A, 12 MISMATCH->B).

Model: Qwen2.5-7B-Instruct, BF16, eval, inference_mode, batch_size=1.
Readout: " A"->362, " B"->425, d_raw = l_A - l_B, teacher-forced pos = prompt_len-1.
Greedy first-token diagnostic must agree with teacher-forced.
No final-reserve data involved.

Output: synthetic_readout_regression.csv
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

R = REPO_ROOT / "d4q1_qwen25_7b_true_prefix_final_confirmation_20260803"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]

def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": label, "reason": why,
         "allowed_final_groups": 196, "quarantined_final_groups": 1,
         "quarantined_group_scored": False, "quarantined_group_hidden_state_read": False,
         "final_configuration_changed": False, "hidden_layer": 18, "hidden_token": "R_end",
         "probe_C": 0.01, "probe_refit_used_dev": False, "probe_refit_used_final": False,
         "activation_intervention_run": False, "mistral_loaded": False,
         "prompt_baselines_run": False}, indent=2), encoding="utf-8")
    sys.exit(1)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()

# continuation token audit
ta = tok.encode(" A", add_special_tokens=False)
tb = tok.encode(" B", add_special_tokens=False)
print("token ids:", ta, tb, "equal len:", len(ta) == len(tb), "distinct:", ta != tb,
      "no UNK:", all(i != tok.unk_token_id for i in ta + tb))
if not (len(ta) == 1 and len(tb) == 1 and ta[0] == ACCEPT_ID and tb[0] == REJECT_ID):
    fail("inheritance_or_execution_invalid", f"continuation ids {ta}/{tb} != frozen 362/425")

SYN = json.loads((D1 / "synthetic_pair_manifest.json").read_text(encoding="utf-8"))
assert len(SYN) == 24

def score(q, ref, cand):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(question=q, reference=ref, candidate=cand)}]
    enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt")
    ids = enc["input_ids"].to("cuda")
    plen = ids.shape[1]
    with torch.inference_mode():
        logits = model(ids).logits
    pos = plen - 1
    lA = logits[0, pos, ACCEPT_ID].item()
    lB = logits[0, pos, REJECT_ID].item()
    d_raw = lA - lB
    pred = "A" if d_raw > 0 else ("B" if d_raw < 0 else "TIE")
    gid = int(logits[0, pos].argmax().item())
    gtok = tok.decode([gid])
    g = gtok.strip()
    gpred = "A" if g == "A" else ("B" if g == "B" else f"OTHER({gtok!r})")
    return lA, lB, d_raw, pred, gid, gtok, gpred

rows = []
for sid, q, ref, cand, exp in SYN:
    lA, lB, d, pred, gid, gtok, gpred = score(q, ref, cand)
    correct = pred == exp
    greedy_agree = (gpred == pred) and correct
    rows.append({"id": sid, "expected_label": exp, "l_A": lA, "l_B": lB, "d_raw": d,
                 "predicted_label": pred, "correct": correct,
                 "greedy_id": gid, "greedy_token": gtok, "greedy_pred": gpred,
                 "greedy_agrees": greedy_agree})
    print(f"{sid} exp={exp} pred={pred} d_raw={d:+.3f} greedy={gtok!r}->{gpred} ok={correct} ga={greedy_agree}")

acc = sum(1 for r in rows if r["correct"]) / 24
accA = sum(1 for r in rows if r["expected_label"] == "A" and r["correct"]) / 12
accB = sum(1 for r in rows if r["expected_label"] == "B" and r["correct"]) / 12
ties = sum(1 for r in rows if r["predicted_label"] == "TIE")
greedy_n = sum(1 for r in rows if r["greedy_agrees"])
medA = sorted(r["d_raw"] for r in rows if r["expected_label"] == "A")[6]
medB = sorted(r["d_raw"] for r in rows if r["expected_label"] == "B")[6]
print(f"\nacc={acc:.3f} A={accA:.3f} B={accB:.3f} ties={ties} greedy={greedy_n}/24 medA={medA:+.3f} medB={medB:+.3f}")

with open(R / "synthetic_readout_regression.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

if acc != 1.0 or accA != 1.0 or accB != 1.0 or ties != 0 or greedy_n != 24 or medA <= 0 or medB >= 0:
    fail("inheritance_or_execution_invalid",
         f"readout regression: acc={acc} A={accA} B={accB} ties={ties} greedy={greedy_n} medA={medA} medB={medB}")
print("Phase 1.1 OK: A/B readout regression 24/24")
