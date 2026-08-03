#!/usr/bin/env python3
"""D3-M Phase 0B: full-forward baseline reproduction on dev 195×4 = 780.

Rebuild four-cell T0 inputs from D0 dev 195 groups; standard monolithic forward
(no hook); compare against D1 four_cell_scores_dev.csv row-by-row.
Gates: 780/780 labels match; l_A/l_B/d_raw within BF16 ULP tolerance; no NaN/tie.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch

import d3m_core as C

D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
M = REPO_ROOT / "d3m_qwen25_7b_monolithic_reference_binding_intervention_20260802"


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (M / "artifacts").mkdir(parents=True, exist_ok=True)
    (M / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
        "monolithic_full_forward_only": True, "segmented_execution_used": False,
        "prefix_cache_used": False, "activation_intervention_run": False,
        "prompt_baselines_run": False, "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


def bf16_ulp(x):
    if x == 0:
        return 2 ** -133
    e = math.floor(math.log2(abs(x)))
    return 2 ** (e - 8)


# D1 reference rows
d1_rows = list(csv.DictReader(open(D1 / "four_cell_scores_dev.csv", encoding="utf-8")))
if len(d1_rows) != 780:
    fail("monolithic_baseline_reproduction_invalid", f"D1 rows={len(d1_rows)}")

# Rebuild dev pairs from D0 (split='dev')
pairs = C.load_swap_pairs("dev")
by_gid = {}
for p in pairs:
    by_gid[p["original_group_id"]] = p
if len(by_gid) != 195:
    fail("monolithic_baseline_reproduction_invalid", f"dev groups={len(by_gid)}")

# compare set alignment with D1 csv
d1_gids = set()
for r in d1_rows:
    d1_gids.add(r["source_group_id"])
if d1_gids != set(by_gid.keys()):
    fail("monolithic_baseline_reproduction_invalid", "dev group id set mismatch with D1")

out = []
label_mismatch = 0
max_ulp = 0.0
max_abs_d = 0.0
n_nan = 0
n_tie = 0
cell_acc = {c: [0, 0] for c in ["OO", "OS", "SO", "SS"]}
for p in pairs:
    gid = p["original_group_id"]
    for cell, ref, cand, exp in C.four_cells(p):
        s = C.score_monolithic(p["q"], ref, cand)
        r1 = next(r for r in d1_rows if r["source_group_id"] == gid and r["cell"] == cell)
        d1A, d1B, d1d = float(r1["l_A"]), float(r1["l_B"]), float(r1["d_raw"])
        ulpA = abs(s["l_A"] - d1A) / max(bf16_ulp(d1A), 1e-12)
        ulpB = abs(s["l_B"] - d1B) / max(bf16_ulp(d1B), 1e-12)
        max_ulp = max(max_ulp, ulpA, ulpB)
        dd = abs(s["d_raw"] - d1d)
        max_abs_d = max(max_abs_d, dd)
        if math.isnan(s["l_A"]) or math.isnan(s["l_B"]):
            n_nan += 1
        if s["predicted_label"] == "TIE":
            n_tie += 1
        is_match = (s["predicted_label"] == r1["predicted_label"])
        if not is_match:
            label_mismatch += 1
        cell_acc[cell][0] += 1
        cell_acc[cell][1] += (1 if is_match else 0)
        out.append({"source_group_id": gid, "cell": cell,
                    "d1_predicted_label": r1["predicted_label"], "our_predicted_label": s["predicted_label"],
                    "label_match": is_match, "d1_l_A": d1A, "d1_l_B": d1B, "d1_d_raw": d1d,
                    "our_l_A": s["l_A"], "our_l_B": s["l_B"], "our_d_raw": s["d_raw"],
                    "l_A_bf16_ulp": round(ulpA, 4), "l_B_bf16_ulp": round(ulpB, 4),
                    "d_raw_abs_diff": round(dd, 6)})
    print(f"  {gid[:12]} scored")

with open(M / "monolithic_baseline_reproduction_rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

print(f"\nrows={len(out)} label_mismatch={label_mismatch} max_bf16_ulp={max_ulp:.4f} "
      f"max_abs_d={max_abs_d:.6f} nan={n_nan} tie={n_tie}")
for c in ["OO", "OS", "SO", "SS"]:
    n, ok = cell_acc[c]
    print(f"  {c}: our label-match rate={ok/n:.4f} (D1 acc ref: "
          f"{sum(1 for r in d1_rows if r['cell']==c and r['correct'])/n:.4f})")

# gates
invalid = (label_mismatch != 0 or max_ulp > 2.0 or n_nan != 0 or n_tie != 0)
label = "monolithic_baseline_reproduction_invalid" if invalid else "monolithic_baseline_reproduction_ok"
print("DECISION:", label)
if invalid:
    fail(label, f"label_mismatch={label_mismatch} max_ulp={max_ulp:.3f} nan={n_nan} tie={n_tie}")

(M / "monolithic_baseline_reproduction_summary.json").write_text(json.dumps({
    "rows": len(out), "label_mismatch": label_mismatch,
    "max_bf16_ulp": float(max_ulp), "max_abs_d_raw": float(max_abs_d),
    "nan": n_nan, "tie": n_tie,
    "cell_label_match_rate": {c: cell_acc[c][1] / cell_acc[c][0] for c in cell_acc},
}, indent=2), encoding="utf-8")
print("Phase 0B OK")
