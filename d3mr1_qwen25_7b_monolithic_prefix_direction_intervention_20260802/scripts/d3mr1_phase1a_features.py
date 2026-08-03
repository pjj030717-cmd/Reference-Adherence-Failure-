#!/usr/bin/env python3
"""D3-M-R1 Phase 1a: train 587 true-truncated prefix features + SS label rescoring.

- For each train group: true-truncated prefix (Question+Reference) forward ->
  L18/R_end hidden state (h_prefix). Verify R_end vs D2-R1 contract.
- Rescore SS cell with full monolithic forward -> y label. Cross-check D3-M/D2-R1.
- Save manifest + features (float32) + labels. Capacity audit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch

import d3mr1_core as C

D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
D3M = REPO_ROOT / "d3m_qwen25_7b_monolithic_reference_binding_intervention_20260802"
R = REPO_ROOT / "d3mr1_qwen25_7b_monolithic_prefix_direction_intervention_20260802"


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
        "monolithic_full_forward_only": True, "prefix_cache_used": False,
        "activation_intervention_run": False, "prompt_baselines_run": False,
        "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


# D3-M / D2-R1 train SS labels for cross-check
d3m_train = {r["source_group_id"]: r for r in
             json.loads((D3M / "train_ss_fullforward.json").read_text(encoding="utf-8"))}
d2r1_train = {r["source_group_id"]: r for r in
              json.loads((D2R1 / "scripts" / "_ss_train_scores.json").read_text(encoding="utf-8"))}

pairs = C.load_swap_pairs("train")
if len(pairs) != 587:
    fail("inheritance_or_data_contract_invalid", f"train pairs={len(pairs)}")

# D2-R1 stored prefix hidden states (h_prefix[17] == layer-18 / L18) for R_end re-verification.
# Their h_prefix array stacks hidden_states[1..28], so index 17 maps to layer 18.
D2R1_HID = D2R1 / "prefix_hidden_states"
C.get_model()
rows = []
r_end_mismatch = 0
max_abs_diff = 0.0
for i, p in enumerate(pairs):
    gid = p["original_group_id"]
    # 1. true-truncated prefix L18/R_end feature (re-extracted independently)
    h, plen = C.prefix_l18_render(p["q"], p["r_s"])
    npz = D2R1_HID / f"train_{gid}.npz"
    if not npz.exists():
        fail("inheritance_or_data_contract_invalid", f"missing D2-R1 npz for {gid}")
    d18 = np.load(npz)["h_prefix"].astype(np.float32)[17]
    md = float(np.max(np.abs(d18 - h)))
    max_abs_diff = max(max_abs_diff, md)
    if md != 0.0:
        r_end_mismatch += 1
    # 2. full monolithic SS label
    s = C.score_monolithic(p["q"], p["r_s"], p["c_s"])
    y = 1 if s["predicted_label"] == "B" else 0
    rows.append({
        "source_group_id": gid, "y": y, "predicted_label": s["predicted_label"],
        "d_raw": s["d_raw"], "l_A": s["l_A"], "l_B": s["l_B"],
        "prefix_len": plen, "r_end_pos": plen - 1,
        "d3m_y": d3m_train.get(gid, {}).get("y"),
        "d2r1_pred": d2r1_train.get(gid, {}).get("predicted_label"),
        "h_prefix": h.astype(np.float32),
    })
    if i % 50 == 0:
        print(f"  {i}/587")

print(f"r_end_mismatch vs D2-R1 stored L18: {r_end_mismatch}, max_abs_diff={max_abs_diff}")
if r_end_mismatch != 0:
    fail("inheritance_or_data_contract_invalid", f"r_end mismatch={r_end_mismatch}, max_abs_diff={max_abs_diff}")

# cross-check labels
agree_d3m = sum(1 for r in rows if r["d3m_y"] == r["y"])
agree_d2r1 = sum(1 for r in rows if r["predicted_label"] == r["d2r1_pred"])
print(f"label agree D3-M: {agree_d3m}/587, agree D2-R1: {agree_d2r1}/587")

# manifest
manifest = {
    "feature_source": "true-truncated prefix (Question+Reference), L18/R_end, monolithic prefix forward",
    "n_train": len(rows),
    "r_end_source": "D2-R1 stored prefix_hidden_states (h_prefix[17] == layer 18) re-verification",
    "r_end_mismatch": r_end_mismatch,
    "max_abs_diff_vs_d2r1": max_abs_diff,
    "label_agree_d3m": agree_d3m, "label_agree_d2r1": agree_d2r1,
    "groups": [{k: (None if isinstance(v, np.ndarray) else v) for k, v in r.items()
                if k != "h_prefix"} for r in rows],
}
(R / "true_prefix_hidden_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

feats = np.stack([r["h_prefix"] for r in rows], axis=0)
np.save(R / "train_prefix_l18_rend.npy", feats.astype(np.float16))
json.dump([r["source_group_id"] for r in rows], open(R / "train_prefix_gids.json", "w"))
labels = np.array([r["y"] for r in rows], dtype=np.int64)
np.save(R / "train_prefix_labels.npy", labels)
json.dump([r["d_raw"] for r in rows], open(R / "train_ss_draw.json", "w"))

# capacity audit
y1 = int(labels.sum())
y0 = int(len(labels) - y1)
with open(R / "train_label_capacity_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["subset", "n", "y1", "y0", "note"])
    w.writeheader()
    w.writerow({"subset": "train(all)", "n": len(labels), "y1": y1, "y0": y0,
                "note": "direction construction uses ALL train (no fit/tune split per D3-M-R1)"})
print(f"train capacity: y1={y1} y0={y0}")
print("Phase 1a OK")
