#!/usr/bin/env python3
"""D3-D Phase 2: fixed prefix risk routing.

- v_prefix = D3-M-R1 V_logit@C=0.01 (frozen).
- z_prefix = v_prefix . (h_true_prefix_R_end - mu_prefix_train).
- t_prefix = median of train z_prefix.
- selected(group) iff z_prefix >= t_prefix, for dev/final.
Selection only depends on Question+Reference prefix, never on Candidate/SS label/
full-prompt score/intervention results. Selected groups get patched on all four cells.

dev selected gates: selected groups >= 70 and baseline SS errors among selected >= 30,
else causal_location_execution_invalid.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch

import d3d_core as C

D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D3MR1 = REPO_ROOT / "d3mr1_qwen25_7b_monolithic_prefix_direction_intervention_20260802"
R = REPO_ROOT / "d3d_qwen25_7b_monolithic_causal_location_scan_20260803"


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


art = np.load(D3MR1 / "frozen_direction_artifact.npz")
v_prefix = art["v_raw"].astype(np.float64)
mu_prefix = art["mu_train"].astype(np.float64)
C.get_model()


def z_prefix_of_group(q, r):
    """true-truncated prefix forward: prefix = ids[:R_end+1], read L18 at last pos."""
    ids, r_end, _, _ = C.build_positions(q, r, C.T0.replace("<answer>", r))
    prefix = ids[: r_end + 1]
    pids = torch.tensor([prefix], device="cuda")
    with torch.inference_mode():
        out = C.get_model()(pids, output_hidden_states=True)
    h = out.hidden_states[18][0, -1].cpu().float().numpy().astype(np.float64)
    return float((h - mu_prefix) @ v_prefix), len(prefix)


# ---- train threshold ----
train_pairs = C.load_swap_pairs("train")
train_z = []
for p in train_pairs:
    z, _ = z_prefix_of_group(p["q"], p["r_s"])
    train_z.append(z)
    if len(train_z) % 200 == 0:
        print(f"  train prefix z {len(train_z)}/587")
train_z = np.array(train_z)
t_prefix = float(np.median(train_z))
print(f"t_prefix = {t_prefix:.4f}  (train n={len(train_z)})")

spec = {
    "v_prefix_source": "D3-M-R1 frozen_direction_artifact.npz (V_logit@C=0.01)",
    "mu_prefix_source": "D3-M-R1 frozen_direction_artifact.npz",
    "z_prefix_formula": "v_prefix . (h_true_prefix_L18_R_end - mu_prefix)",
    "t_prefix": t_prefix,
    "t_prefix_calculation": "median of train z_prefix (587 groups)",
    "selection_rule": "selected(group) iff z_prefix >= t_prefix",
    "note": "selection only reads Question+Reference prefix; never Candidate/SS label/full prompt",
}
(R / "frozen_prefix_risk_selection_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")

# ---- dev routing ----
d1 = {f"{r['source_group_id']}|{r['cell']}": r for r in
      csv.DictReader(open(D1 / "four_cell_scores_dev.csv", encoding="utf-8"))}
dev_pairs = C.load_swap_pairs("dev")
dev_rows = []
for p in dev_pairs:
    z, plen = z_prefix_of_group(p["q"], p["r_s"])
    selected = int(z >= t_prefix)
    dev_rows.append({"source_group_id": p["original_group_id"], "z_prefix": z,
                     "prefix_len": plen, "selected": selected,
                     "ss_d1_pred": d1[f"{p['original_group_id']}|SS"]["predicted_label"],
                     "ss_d1_correct": d1[f"{p['original_group_id']}|SS"]["correct"]})
dev_rows.sort(key=lambda r: -r["z_prefix"])
with open(R / "dev_selected_group_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(dev_rows[0].keys()))
    w.writeheader()
    w.writerows(dev_rows)

n_sel = sum(1 for r in dev_rows if r["selected"])
ss_base_sel = sum(1 for r in dev_rows if r["selected"] and r["ss_d1_correct"] == "False")
print(f"dev selected={n_sel}, selected SS baseline errors={ss_base_sel}")
if n_sel < 70 or ss_base_sel < 30:
    fail("causal_location_execution_invalid",
         f"dev selected={n_sel} (<70) or SS base errors={ss_base_sel} (<30)")

print("Phase 2 OK")
