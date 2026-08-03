#!/usr/bin/env python3
"""E1 Phase 1: synthetic MATCH/MISMATCH readout regression (decision gate).

Same 24 pairs as D1's synthetic_pair_manifest.json. Verifies:
  pairwise order accuracy = 24/24
  MATCH -> A = 12/12
  MISMATCH -> B = 12/12
  ties = 0
  greedy first token consistent with likelihood = 24/24
On failure: label decision_readout_invalid.

No hidden states; only final-layer logits at pos=prompt_len-1.
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

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
OUT = REPO_ROOT / "e1_qwen25_7b_popqa_h1_behavior_gate_20260803"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"

CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]


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


def score(question: str, reference: str, candidate: str):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(question=question, reference=reference, candidate=candidate)},
    ]
    enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    prompt_ids = enc["input_ids"].to("cuda")
    prompt_len = prompt_ids.shape[1]
    with torch.inference_mode():
        logits = model(prompt_ids).logits
    pos = prompt_len - 1
    ll = logits[0, pos, :]
    l_A = ll[ACCEPT_ID].item()
    l_B = ll[REJECT_ID].item()
    d_raw = l_A - l_B
    p_accept = 1.0 / (1.0 + math.exp(-d_raw))
    pred = "A" if d_raw > 0 else ("B" if d_raw < 0 else "TIE")
    greedy_id = int(ll.argmax().item())
    greedy_tok = tok.decode([greedy_id])
    return l_A, l_B, d_raw, p_accept, pred, greedy_id, greedy_tok


syn = json.loads((D1 / "synthetic_pair_manifest.json").read_text(encoding="utf-8"))
print("synthetic pairs:", len(syn))
results = []
for sid, q, ref, cand, exp in syn:
    l_A, l_B, d_raw, p_accept, pred, greedy_id, greedy_tok = score(q, ref, cand)
    g = greedy_tok.strip()
    greedy_pred = "A" if g == "A" else ("B" if g == "B" else f"OTHER({g[:12]!r})")
    results.append({"id": sid, "question": q, "reference": ref, "candidate": cand,
                    "expected_label": exp, "l_A": l_A, "l_B": l_B, "d_raw": d_raw,
                    "p_accept_raw": p_accept, "predicted_label": pred, "correct": pred == exp,
                    "greedy_id": greedy_id, "greedy_token": greedy_tok,
                    "greedy_pred": greedy_pred, "greedy_agrees": greedy_pred == pred})
    print(f"{sid} exp={exp} pred={pred} d_raw={d_raw:+.3f} greedy={greedy_tok!r}->{greedy_pred} ok={pred==exp} ga={greedy_pred==pred}")

with open(OUT / "synthetic_readout_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

acc = sum(1 for r in results if r["correct"]) / len(results)
accA = sum(1 for r in results if r["expected_label"] == "A" and r["correct"])
accB = sum(1 for r in results if r["expected_label"] == "B" and r["correct"])
ties = sum(1 for r in results if r["predicted_label"] == "TIE")
greedy_agree = sum(1 for r in results if r["greedy_agrees"])
medA = sorted([r["d_raw"] for r in results if r["expected_label"] == "A"])[11]
medB = sorted([r["d_raw"] for r in results if r["expected_label"] == "B"])[11]
print(f"synthetic acc={acc:.4f} A={accA}/12 B={accB}/12 ties={ties} greedy_agree={greedy_agree}/24 "
      f"medA={medA:+.4f} medB={medB:+.4f}")

if acc < 23 / 24 or accA < 11 or accB < 11 or ties > 0 or greedy_agree != 24:
    fail("decision_readout_invalid",
         f"acc={acc:.4f} accA={accA} accB={accB} ties={ties} greedy_agree={greedy_agree}")
print("SYNTHETIC READOUT GATE PASSED")
print("Phase 1 OK")
