#!/usr/bin/env python3
"""E01-D1-L Phase 1: synthetic readout regression + T0 exact reproduction.

Loads Qwen2.5-7B-Instruct (same revision, BF16, eval, inference_mode, batch=1).
1) 24 synthetic MATCH/MISMATCH pairs from D1's synthetic_pair_manifest.json:
   - pairwise order accuracy 24/24, MATCH->A 12/12, MISMATCH->B 12/12, ties=0,
     greedy first token consistent with likelihood 24/24.
   On failure: label decision_readout_invalid.
2) T0 dev exact reproduction: 195 groups x 4 cells = 780, compared row-by-row
   with D1 four_cell_scores_dev.csv (pred labels 780/780; l_A/l_B/d_raw identical).
   Aggregate: OO/OS/SO/SS acc = 1.000/1.000/0.928/0.241; SS FR = 0.759.
   On failure: label baseline_reproduction_invalid.

No hidden states; only final-layer logits at pos = prompt_len-1 for the A/B tokens.
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
OUT = REPO_ROOT / "d1l_qwen25_7b_long_candidate_expression_robustness_20260803"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"

CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]


def fail(label, why):
    print("STOP:", label, "-", why)
    (OUT / "artifacts").mkdir(parents=True, exist_ok=True)
    (OUT / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "hidden_states_read": False, "probe_trained": False,
        "activation_intervention_run": False, "final_reserve_model_scored": False,
        "train_text_read": False, "final_reserve_text_read": False,
        "model_prompt_changed": False}, indent=2), encoding="utf-8")
    sys.exit(1)


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()
print("model loaded for Phase 1 (synthetic + T0)")


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


# ================= 1. synthetic readout regression =================
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
nA = sum(1 for r in results if r["expected_label"] == "A")
nB = sum(1 for r in results if r["expected_label"] == "B")
accA = sum(1 for r in results if r["expected_label"] == "A" and r["correct"])
accB = sum(1 for r in results if r["expected_label"] == "B" and r["correct"])
ties = sum(1 for r in results if r["predicted_label"] == "TIE")
greedy_agree = sum(1 for r in results if r["greedy_agrees"])
medA = sorted([r["d_raw"] for r in results if r["expected_label"] == "A"])[nA // 2]
medB = sorted([r["d_raw"] for r in results if r["expected_label"] == "B"])[nB // 2]
print(f"synthetic acc={acc:.4f} A={accA}/12 B={accB}/12 ties={ties} greedy_agree={greedy_agree}/24 "
      f"medA={medA:+.4f} medB={medB:+.4f}")

if acc < 23 / 24 or accA < 11 or accB < 11 or ties > 0 or greedy_agree != 24:
    fail("decision_readout_invalid",
         f"acc={acc:.4f} accA={accA} accB={accB} ties={ties} greedy_agree={greedy_agree}")
print("SYNTHETIC READOUT GATE PASSED")

# ================= 2. T0 exact reproduction =================
spec = json.loads((OUT / "candidate_length_expression_spec.json").read_text(encoding="utf-8"))
T0 = spec["templates"]["T0"]["template"]
dev_pairs = []
with open(OUT / "scripts" / "_dev_input.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dev_pairs.append(json.loads(line))
print("dev pairs:", len(dev_pairs))


def render(tpl, ans):
    return tpl.replace("<answer>", ans)


t0_rows = []
for p in dev_pairs:
    gid = p["original_group_id"]
    specs = [
        ("OO", p["r_o"], render(T0, p["r_o"])),
        ("OS", p["r_o"], render(T0, p["r_s"])),
        ("SO", p["r_s"], render(T0, p["r_o"])),
        ("SS", p["r_s"], render(T0, p["r_s"])),
    ]
    for cell, ref, cand in specs:
        l_A, l_B, d_raw, p_accept, pred, _, _ = score(p["q"], ref, cand)
        exp = "A" if cell in ("OO", "SS") else "B"
        t0_rows.append({"source_group_id": gid, "cell": cell, "question": p["q"],
                        "reference": ref, "candidate": cand, "expected_label": exp,
                        "l_A": l_A, "l_B": l_B, "d_raw": d_raw, "p_accept_raw": p_accept,
                        "predicted_label": pred, "correct": pred == exp})
print("T0 rows:", len(t0_rows))

with open(OUT / "t0_reproduction_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(t0_rows[0].keys()))
    w.writeheader()
    w.writerows(t0_rows)

# ---- compare with D1 four_cell_scores_dev.csv ----
d1_map = {}
with open(D1 / "four_cell_scores_dev.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d1_map[(r["source_group_id"], r["cell"])] = r

mismatches = []
label_mismatch = 0
for r in t0_rows:
    d1r = d1_map.get((r["source_group_id"], r["cell"]))
    if d1r is None:
        mismatches.append((r["source_group_id"], r["cell"], "missing in D1"))
        continue
    if r["predicted_label"] != d1r["predicted_label"]:
        label_mismatch += 1
        mismatches.append((r["source_group_id"], r["cell"],
                           f"label {r['predicted_label']} vs D1 {d1r['predicted_label']}"))
        continue
    for fld in ("l_A", "l_B", "d_raw"):
        if float(r[fld]) != float(d1r[fld]):
            mismatches.append((r["source_group_id"], r["cell"],
                               f"{fld} {r[fld]:.6f} vs D1 {d1r[fld]:.6f}"))

if mismatches:
    print("T0 mismatches:", len(mismatches), "label mismatches:", label_mismatch)
    for m in mismatches[:10]:
        print("  ", m)
    fail("baseline_reproduction_invalid", f"{len(mismatches)} mismatches vs D1 (labels {label_mismatch})")

# aggregate accuracy
cells = {}
for c in ["OO", "OS", "SO", "SS"]:
    sub = [r for r in t0_rows if r["cell"] == c]
    cells[c] = sum(1 for r in sub if r["correct"]) / len(sub)
print("T0 aggregate:", {c: round(v, 3) for c, v in cells.items()})
fr_ss = sum(1 for r in t0_rows if r["cell"] == "SS" and r["predicted_label"] == "B") / 195
exp_cells = {"OO": 1.0, "OS": 1.0, "SO": 0.928, "SS": 0.241}
if any(abs(cells[c] - exp_cells[c]) > 5e-4 for c in exp_cells):
    fail("baseline_reproduction_invalid", f"aggregate mismatch {cells}")
if abs(fr_ss - 0.759) > 5e-3:
    fail("baseline_reproduction_invalid", f"SS FR {fr_ss:.4f} != 0.759")
print("T0 EXACT REPRODUCTION OK (780/780, bitwise l_A/l_B/d_raw, aggregates matched)")

# T0 SS error groups (for retention anchors)
t0_ss_err = {r["source_group_id"] for r in t0_rows if r["cell"] == "SS" and not r["correct"]}
print("T0 SS error groups:", len(t0_ss_err))
(OUT / "scripts" / "_t0_ss_error_groups.json").write_text(json.dumps(sorted(t0_ss_err)), encoding="utf-8")
print("Phase 1 OK")
