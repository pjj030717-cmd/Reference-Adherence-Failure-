#!/usr/bin/env python3
"""D3-D Phase 3: dev causal location scan (12 locations x 6 alpha) +
specificity controls (Z/R/V/N1-5) on the single best config.

Scan on selected groups only (fixed prefix risk routing, Phase 2).
Metrics: SS_net_gain, nonSS_added_harm, CSI. Admission:
  selected_SS_base_error_count>=30, SS_net_gain>=0.15, nonSS_added_harm<=0.02, CSI>=0.13.
Selection rule: max CSI; tie SS_net_gain; tie |alpha|; tie position C_end>D_pos>R_end;
tie layer L18>L22>L14>L26.
Specificity: CSI_real>0, CSI_real>CSI_reverse, CSI_real>median(CSI_random_1..5).
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
R = REPO_ROOT / "d3d_qwen25_7b_monolithic_causal_location_scan_20260803"

ALPHAS = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]
POS_PRIORITY = {"C_end": 0, "D_pos": 1, "R_end": 2}
LAYER_PRIORITY = {18: 0, 22: 1, 14: 2, 26: 3}
RANDOM_SEED = 20260808


def stop(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
        "monolithic_full_forward_only": True, "prefix_cache_used": False,
        "activation_intervention_run": True, "prompt_baselines_run": False,
        "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


# load frozen directions
sel = json.loads((R / "train_location_direction_selection.json").read_text(encoding="utf-8"))
art = np.load(R / "train_location_direction_artifact.npz")
def load_loc(loc):
    k = loc.replace("/", "_")
    v = art[f"{k}_v"].astype(np.float64)
    mu = art[f"{k}_mu"].astype(np.float64)
    sigma = float(art[f"{k}_sigma"][0])
    return v, mu, sigma

# selected dev groups
dev_rows = list(csv.DictReader(open(R / "dev_selected_group_manifest.csv", encoding="utf-8")))
sel_ids = {r["source_group_id"] for r in dev_rows if r["selected"] == "1"}
print(f"selected groups: {len(sel_ids)}")

# base errors from D1
d1 = {f"{r['source_group_id']}|{r['cell']}": r for r in
      csv.DictReader(open(D1 / "four_cell_scores_dev.csv", encoding="utf-8"))}
base_err = {k: (1 if r["correct"] == "False" else 0) for k, r in d1.items()}

pairs = C.load_swap_pairs("dev")
sel_pairs = [p for p in pairs if p["original_group_id"] in sel_ids]
ss_base_sel = sum(1 for p in sel_pairs if d1[f"{p['original_group_id']}|SS"]["correct"] == "False")
print(f"selected SS base errors: {ss_base_sel}")
if ss_base_sel < 30:
    stop("causal_location_execution_invalid", f"selected SS base errors={ss_base_sel}<30")

C.get_model()
LOCS = [f"L{li}/{pos}" for li in C.CAND_LAYERS for pos in C.POSITIONS]

def make_apply(v, mu, sigma, alpha):
    delta = alpha * sigma * v
    dt = torch.tensor(delta, dtype=torch.float32, device="cuda")
    def apply_fn(hidden):
        return hidden.to(torch.float32) + dt
    return apply_fn

def eval_config(loc, alpha):
    v, mu, sigma = load_loc(loc)
    li = int(loc.split("/")[0][1:])
    pos = loc.split("/")[1]
    ss_base = ss_patch = nonSS_patch = nonSS_base = n_cells = 0
    n_ss = 0
    for p in sel_pairs:
        n_ss += 1
        for cell, ref, cand, exp in C.four_cells(p):
            key = f"{p['original_group_id']}|{cell}"
            eb = base_err[key]
            if cell == "SS":
                ss_base += eb
            s = C.run_intervention(p["q"], ref, cand, li, pos, apply_fn=make_apply(v, mu, sigma, alpha))
            ep = 1 if s["predicted_label"] != exp else 0
            if cell == "SS":
                ss_patch += ep
            else:
                nonSS_patch += ep
                nonSS_base += eb
                n_cells += 1
    SS_net_gain = (ss_base - ss_patch) / n_ss
    nonSS_added_harm = (nonSS_patch - nonSS_base) / n_cells if n_cells else 0.0
    CSI = SS_net_gain - nonSS_added_harm
    return {"location": loc, "layer": li, "position": pos, "alpha": alpha,
            "selected_SS_base_error_count": ss_base, "SS_net_gain": round(SS_net_gain, 6),
            "nonSS_added_harm": round(nonSS_added_harm, 6), "CSI": round(CSI, 6),
            "qualified": int(ss_base >= 30 and SS_net_gain >= 0.15 and nonSS_added_harm <= 0.02 and CSI >= 0.13)}

scan_rows = []
for loc in LOCS:
    for alpha in ALPHAS:
        r = eval_config(loc, alpha)
        scan_rows.append(r)
        print(f"  {loc} alpha={alpha}: base_SS={r['selected_SS_base_error_count']} gain={r['SS_net_gain']:.4f} harm={r['nonSS_added_harm']:.4f} CSI={r['CSI']:.4f} qlf={r['qualified']}")

with open(R / "dev_location_alpha_scan.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
    w.writeheader()
    w.writerows(scan_rows)

qualified = [r for r in scan_rows if r["qualified"]]
if not qualified:
    stop("causal_location_dev_not_found", f"{len(scan_rows)} configs scanned, 0 qualified")

best = sorted(qualified, key=lambda r: (-r["CSI"], -r["SS_net_gain"], abs(r["alpha"]),
                                        POS_PRIORITY[r["position"]], LAYER_PRIORITY[r["layer"]]))[0]
print(f"best: {best['location']} alpha={best['alpha']} CSI={best['CSI']} gain={best['SS_net_gain']} harm={best['nonSS_added_harm']}")

# ---- specificity controls on best config ----
v, mu, sigma = load_loc(best["location"])
li = best["layer"]
pos = best["position"]
alpha_star = best["alpha"]

# random directions
rng = np.random.default_rng(RANDOM_SEED)
rand_dirs = []
for k in range(5):
    d = rng.standard_normal(3584)
    d = d - (d @ v) * v
    d = d / (np.linalg.norm(d) + 1e-12)
    rand_dirs.append(d)

def run_cond(delta_vec):
    dt = torch.tensor(delta_vec, dtype=torch.float32, device="cuda")
    def apply_fn(hidden):
        return hidden.to(torch.float32) + dt
    return apply_fn

def cond_metrics(apply_fn):
    ss_base = ss_patch = nonSS_patch = nonSS_base = n_cells = n_ss = 0
    for p in sel_pairs:
        n_ss += 1
        for cell, ref, cand, exp in C.four_cells(p):
            key = f"{p['original_group_id']}|{cell}"
            eb = base_err[key]
            if cell == "SS":
                ss_base += eb
            s = C.run_intervention(p["q"], ref, cand, li, pos, apply_fn=apply_fn)
            ep = 1 if s["predicted_label"] != exp else 0
            if cell == "SS":
                ss_patch += ep
            else:
                nonSS_patch += ep
                nonSS_base += eb
                n_cells += 1
    gain = (ss_base - ss_patch) / n_ss
    harm = (nonSS_patch - nonSS_base) / n_cells if n_cells else 0.0
    return gain, harm, gain - harm

zero_apply = None
def zero_fn(hidden):
    return hidden.to(torch.float32)

controls = []
g_r, h_r, c_r = cond_metrics(run_cond(alpha_star * sigma * v))
controls.append({"condition": "R_real", "SS_net_gain": round(g_r, 6), "nonSS_added_harm": round(h_r, 6), "CSI": round(c_r, 6)})
g_v, h_v, c_v = cond_metrics(run_cond(-alpha_star * sigma * v))
controls.append({"condition": "V_reverse", "SS_net_gain": round(g_v, 6), "nonSS_added_harm": round(h_v, 6), "CSI": round(c_v, 6)})
rand_csi = []
for k, d in enumerate(rand_dirs, 1):
    g, h, c = cond_metrics(run_cond(alpha_star * sigma * d))
    rand_csi.append(c)
    controls.append({"condition": f"N{k}_random", "SS_net_gain": round(g, 6), "nonSS_added_harm": round(h, 6), "CSI": round(c, 6)})
with open(R / "dev_specificity_controls.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "SS_net_gain", "nonSS_added_harm", "CSI"])
    w.writeheader()
    w.writerows(controls)

csi_real = c_r
csi_rev = c_v
csi_rand_med = float(np.median(rand_csi))
print(f"specificity: CSI_real={csi_real:.4f} CSI_reverse={csi_rev:.4f} median(random)={csi_rand_med:.4f}")
if not (csi_real > 0 and csi_real > csi_rev and csi_real > csi_rand_med):
    stop("causal_location_dev_not_found",
         f"specificity failed: CSI_real={csi_real:.4f} rev={csi_rev:.4f} med_rand={csi_rand_med:.4f}")

# freeze
freeze = {
    "selected_location": best["location"],
    "selected_layer": li,
    "selected_position": pos,
    "direction_method": sel["direction_method_per_location"][best["location"]],
    "alpha_star": alpha_star,
    "t_prefix": json.loads((R / "frozen_prefix_risk_selection_spec.json").read_text(encoding="utf-8"))["t_prefix"],
    "CSI_real": csi_real, "CSI_reverse": csi_rev, "CSI_median_random": csi_rand_med,
    "dev_metrics": scan_rows,
    "random_seed_dev": RANDOM_SEED,
    "frozen_before_final_reserve": True,
}
(R / "dev_configuration_freeze.json").write_text(json.dumps(freeze, indent=1), encoding="utf-8")
print(f"FROZEN: {best['location']} alpha*={alpha_star} CSI={csi_real:.4f}")
print("Phase 3 OK")
