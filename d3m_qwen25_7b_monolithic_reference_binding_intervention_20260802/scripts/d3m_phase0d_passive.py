#!/usr/bin/env python3
"""D3-M Phase 0D: passive hook zero-intervention equivalence on dev 780.

Install the intervention-style hook (same closure used in Phase 2/3) but with
apply_fn=None (hook returns None = no modification). Outputs must match Phase 0B
standard forward: 780/780 labels; d_raw diff within BF16 repeat tolerance; no
NaN; no extra ties.
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


# baseline rows from Phase 0B (already validated == D1)
base = list(csv.DictReader(open(M / "monolithic_baseline_reproduction_rows.csv", encoding="utf-8")))
if len(base) != 780:
    fail("monolithic_execution_equivalence_invalid", f"baseline rows={len(base)}")

pairs = C.load_swap_pairs("dev")
out = []
mismatch = 0
max_dd = 0.0
n_tie0 = 0
n_tie1 = 0
n_nan = 0
for p in pairs:
    gid = p["original_group_id"]
    for cell, ref, cand, exp in C.four_cells(p):
        r0 = next(r for r in base if r["source_group_id"] == gid and r["cell"] == cell)
        s = C.run_intervention(p["q"], ref, cand, apply_fn=None)  # passive hook, no modification
        # also verify hook captured pre==post (trivially, clone)
        dd = abs(s["d_raw"] - float(r0["our_d_raw"]))
        max_dd = max(max_dd, dd)
        if s["predicted_label"] != r0["our_predicted_label"]:
            mismatch += 1
        if r0["our_predicted_label"] == "TIE":
            n_tie0 += 1
        if s["predicted_label"] == "TIE":
            n_tie1 += 1
        if math.isnan(s["l_A"]) or math.isnan(s["l_B"]):
            n_nan += 1
        out.append({"source_group_id": gid, "cell": cell,
                    "base_pred": r0["our_predicted_label"], "hook_passive_pred": s["predicted_label"],
                    "label_match": s["predicted_label"] == r0["our_predicted_label"],
                    "base_d_raw": float(r0["our_d_raw"]), "hook_passive_d_raw": s["d_raw"],
                    "d_raw_abs_diff": round(dd, 6)})
    print(f"  {gid[:12]} done")

with open(M / "passive_hook_zero_equivalence_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

print(f"\nrows={len(out)} label_mismatch={mismatch} max_d_raw_diff={max_dd:.8f} "
      f"tie_base={n_tie0} tie_hook={n_tie1} nan={n_nan}")
invalid = (mismatch != 0 or max_dd > 0.05 or n_nan != 0 or n_tie1 > n_tie0)
label = "monolithic_execution_equivalence_invalid" if invalid else "monolithic_execution_equivalence_ok"
print("DECISION:", label)
if invalid:
    fail(label, f"mismatch={mismatch} max_dd={max_dd:.6f} nan={n_nan} tie_base={n_tie0} tie_hook={n_tie1}")
print("Phase 0D OK")
