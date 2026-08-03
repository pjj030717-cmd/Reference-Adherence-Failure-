#!/usr/bin/env python3
"""D3-M-R1 Phase 2: dev risk selection + (q x alpha) grid, freeze unique config.

2.1: dev 195 true-truncated prefix -> L18/R_end hidden -> z_dev = v*.(h - mu_train).
     coverage q in {1.00, 0.75, 0.50, 0.25}: top-q groups by z_dev.
2.2: monolithic full-forward intervention at layers[17] output / R_end:
     h_patched = h + alpha * sigma_z_train * v*, alpha in 8 values.
2.3: metrics on selected groups: SS_net_gain, nonSS_added_harm, CSI.
     Config qualifies iff selected_SS_base_error_count>=20, SS_net_gain>=0.10,
     nonSS_added_harm<=0.02, CSI>=0.08. Choose max CSI, tie->SS_net_gain,
     tie->|alpha| smaller, tie->q smaller. Freeze.
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

import d3mr1_core as C

D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
R = REPO_ROOT / "d3mr1_qwen25_7b_monolithic_prefix_direction_intervention_20260802"

ALPHAS = [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]
QS = [1.00, 0.75, 0.50, 0.25]


def stop(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
        "monolithic_full_forward_only": True, "prefix_cache_used": False,
        "activation_intervention_run": False, "prompt_baselines_run": False,
        "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


# frozen direction
md = json.loads((R / "frozen_direction_metadata.json").read_text(encoding="utf-8"))
art = np.load(R / "frozen_direction_artifact.npz")
v_raw = art["v_raw"].astype(np.float64)
mu_train = art["mu_train"].astype(np.float64)
sigma_z = float(art["sigma_z_train"])

# ---- 2.1 dev risk selection ----
d1_rows = {f"{r['source_group_id']}|{r['cell']}": r for r in
           csv.DictReader(open(D1 / "four_cell_scores_dev.csv", encoding="utf-8"))}
pairs = C.load_swap_pairs("dev")
assert len(pairs) == 195

C.get_model()
rows = []
D2R1_HID = D2R1 / "prefix_hidden_states"
for p in pairs:
    gid = p["original_group_id"]
    h, plen = C.prefix_l18_render(p["q"], p["r_s"])
    d18 = np.load(D2R1_HID / f"dev_{gid}.npz")["h_prefix"].astype(np.float32)[17]
    assert float(np.max(np.abs(d18 - h))) == 0.0, f"prefix mismatch {gid}"
    z_dev = float((h.astype(np.float64) - mu_train) @ v_raw)
    rows.append({"source_group_id": gid, "z_dev": z_dev, "prefix_len": plen})
rows.sort(key=lambda r: -r["z_dev"])


def _patch(delta):
    delta_t = torch.tensor(delta * v_raw, dtype=torch.float32, device="cuda")

    def apply_fn(hidden):
        return hidden.to(torch.float32) + delta_t
    return apply_fn

# per-cell base error from D1
base_err = {f"{r['source_group_id']}|{r['cell']}": (1 if r["correct"] == "False" else 0)
            for r in d1_rows.values()}

grid_rows = []
qualified = []
for q in QS:
    n_sel = int(round(195 * q))
    sel = rows[:n_sel]
    sel_ids = {r["source_group_id"] for r in sel}
    for alpha in ALPHAS:
        delta = alpha * sigma_z
        ss_err_base = ss_err_patch = nonSS_patch = nonSS_base = n_cells = 0
        n_ss = 0
        for p in pairs:
            gid = p["original_group_id"]
            if gid not in sel_ids:
                continue
            n_ss += 1
            for cell, ref, cand, exp in C.four_cells(p):
                key = f"{gid}|{cell}"
                eb = base_err[key]
                if cell == "SS":
                    ss_err_base += eb
                s = C.run_intervention(p["q"], ref, cand, apply_fn=_patch(delta))
                ep = 1 if s["predicted_label"] != exp else 0
                if cell == "SS":
                    ss_err_patch += ep
                else:
                    nonSS_patch += ep
                    nonSS_base += eb
                    n_cells += 1
        SS_net_gain = (ss_err_base - ss_err_patch) / n_ss
        nonSS_added_harm = (nonSS_patch - nonSS_base) / n_cells if n_cells else 0.0
        CSI = SS_net_gain - nonSS_added_harm
        row = {"q": q, "alpha": alpha, "n_selected": n_sel,
               "selected_SS_base_error_count": ss_err_base,
               "SS_net_gain": round(SS_net_gain, 6),
               "nonSS_added_harm": round(nonSS_added_harm, 6),
               "CSI": round(CSI, 6),
               "qualified": 1 if (ss_err_base >= 20 and SS_net_gain >= 0.10
                                  and nonSS_added_harm <= 0.02 and CSI >= 0.08) else 0}
        grid_rows.append(row)
        if row["qualified"]:
            qualified.append(row)
        print(f"  q={q} alpha={alpha} base_err_SS={ss_err_base} gain={SS_net_gain:.4f} harm={nonSS_added_harm:.4f} CSI={CSI:.4f}")

with open(R / "dev_risk_selection_and_grid.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
    w.writeheader()
    w.writerows(grid_rows)

# risk selection sheet
with open(R / "dev_risk_selection_and_grid.csv", "a", newline="") as f:
    f.write("\n# dev risk scores (top-q by z_dev)\n")
    f.write("source_group_id,z_dev,prefix_len\n")
    for r in rows:
        f.write(f"{r['source_group_id']},{r['z_dev']},{r['prefix_len']}\n")

if not qualified:
    stop("monolithic_patch_dev_selectivity_insufficient",
         f"{len(grid_rows)} configs evaluated, 0 qualified")

# selection: max CSI, tie->SS_net_gain, tie->|alpha|, tie->q
best = sorted(qualified, key=lambda r: (-r["CSI"], -r["SS_net_gain"],
                                        abs(r["alpha"]), r["q"]))[0]

freeze = {
    "direction_method": md["direction_method"],
    "v_raw_sha256": None,
    "mu_train": mu_train.tolist(),
    "sigma_z_train": sigma_z,
    "alpha_star": best["alpha"],
    "q_star": best["q"],
    "selection_rule": "CSI on dev; only groups in top-q by z_dev patched; all 4 cells of selected groups patched",
    "threshold": "selected_SS_base_error_count>=20; SS_net_gain>=0.10; nonSS_added_harm<=0.02; CSI>=0.08",
    "selected_config": best,
    "hook": "forward_hook on model.model.layers[17] output, position R_end, h+alpha*sigma_z*v*",
    "dev_metrics": grid_rows,
    "frozen_before_final_reserve": True,
}
(R / "dev_configuration_freeze.json").write_text(json.dumps(freeze, indent=1), encoding="utf-8")
print(f"FROZEN: q*={best['q']} alpha*={best['alpha']} CSI={best['CSI']} gain={best['SS_net_gain']} harm={best['nonSS_added_harm']}")
print("Phase 2 OK")
