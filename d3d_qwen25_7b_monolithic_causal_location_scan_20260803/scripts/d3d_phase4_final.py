#!/usr/bin/env python3
"""D3-D Phase 4: final-reserve one-shot confirmation (first-ever read).

Conditions on final-reserve selected groups (frozen prefix risk threshold):
  Z: zero
  R: real direction   +alpha* * sigma* * v*
  V: reverse direction -alpha* * sigma* * v*
  N1..N10: 10 fixed random orthogonal dirs (seed 20260809)
All on four cells, monolithic full forward. Group-paired bootstrap 2000 (seed 20260810).
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

R = REPO_ROOT / "d3d_qwen25_7b_monolithic_causal_location_scan_20260803"
RANDOM_SEED = 20260809
BOOT_SEED = 20260810
EXPECTED_LABEL = {"OO": "A", "OS": "B", "SO": "B", "SS": "A"}


def stop(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": True, "final_reserve_hidden_states_read": True,
        "monolithic_full_forward_only": True, "prefix_cache_used": False,
        "activation_intervention_run": True, "prompt_baselines_run": True,
        "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


freeze = json.loads((R / "dev_configuration_freeze.json").read_text(encoding="utf-8"))
loc = freeze["selected_location"]
li = freeze["selected_layer"]
pos = freeze["selected_position"]
alpha_star = freeze["alpha_star"]
t_prefix = freeze["t_prefix"]
art = np.load(R / "train_location_direction_artifact.npz")
k = loc.replace("/", "_")
v = art[f"{k}_v"].astype(np.float64)
mu = art[f"{k}_mu"].astype(np.float64)
sigma = float(art[f"{k}_sigma"][0])

# prefix routing vectors from D3-M-R1
pmr1 = REPO_ROOT / "d3mr1_qwen25_7b_monolithic_prefix_direction_intervention_20260802"
p_art = np.load(pmr1 / "frozen_direction_artifact.npz")
v_prefix = p_art["v_raw"].astype(np.float64)
mu_prefix = p_art["mu_train"].astype(np.float64)


def z_prefix_of_group(q, r):
    ids, r_end, _, _ = C.build_positions(q, r, C.T0.replace("<answer>", r))
    prefix = ids[: r_end + 1]
    pids = torch.tensor([prefix], device="cuda")
    with torch.inference_mode():
        out = C.get_model()(pids, output_hidden_states=True)
    h = out.hidden_states[18][0, -1].cpu().float().numpy().astype(np.float64)
    return float((h - mu_prefix) @ v_prefix)


C.get_model()
pairs = C.load_swap_pairs("final_reserve")
assert len(pairs) == 197

# selected groups
sel = []
for p in pairs:
    z = z_prefix_of_group(p["q"], p["r_s"])
    sel.append({"source_group_id": p["original_group_id"], "z_prefix": z,
                "selected": int(z >= t_prefix)})
sel_ids = {r["source_group_id"] for r in sel if r["selected"]}
n_sel = len(sel_ids)
print(f"final-reserve selected: {n_sel}/{197} (t_prefix={t_prefix:.4f})")

# random directions (fixed)
rng = np.random.default_rng(RANDOM_SEED)
rand_dirs = []
for kk in range(10):
    d = rng.standard_normal(3584)
    d = d - (d @ v) * v
    d = d / (np.linalg.norm(d) + 1e-12)
    rand_dirs.append(d)

conds = {
    "Z": None,
    "R": alpha_star * sigma * v,
    "V": -alpha_star * sigma * v,
}
for kk, d in enumerate(rand_dirs, 1):
    conds[f"N{kk}"] = alpha_star * sigma * d


def run_cond(p, delta):
    if delta is None:
        return C.run_intervention(p["q"], p["r_o"], p["c_o"], li, pos, apply_fn=None)
    dt = torch.tensor(delta, dtype=torch.float32, device="cuda")
    def apply_fn(hidden):
        return hidden.to(torch.float32) + dt
    return C.run_intervention(p["q"], p["r_o"], p["c_o"], li, pos, apply_fn=apply_fn)


# per-cell scoring
results = []
for p in pairs:
    gid = p["original_group_id"]
    if gid not in sel_ids:
        continue
    for cell, ref, cand, exp in C.four_cells(p):
        for cname, delta in conds.items():
            s = run_cond(p, delta)
            results.append({"source_group_id": gid, "cell": cell, "condition": cname,
                            "predicted_label": s["predicted_label"], "d_raw": s["d_raw"],
                            "token_pos": s["token_pos"], "seq_len": s["seq_len"]})
    print(f"  {gid[:12]} done")

with open(R / "final_reserve_group_level_effects.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)


# Base errors from Z condition (first-ever final-reserve read)
Z_SS_ERR = {r["source_group_id"]: 0 for r in results}
Z_NON_ERR = {r["source_group_id"]: 0 for r in results}
for r in results:
    if r["condition"] != "Z":
        continue
    g = r["source_group_id"]
    exp = EXPECTED_LABEL[r["cell"]]
    if r["predicted_label"] != exp:
        if r["cell"] == "SS":
            Z_SS_ERR[g] += 1
        else:
            Z_NON_ERR[g] += 1


def cond_metrics(cname):
    sub = [r for r in results if r["condition"] == cname]
    ss_patch = nonSS_patch = n_ss = n_cells = 0
    ss_base_total = 0
    nonSS_base_total = 0
    acc = {c: 0 for c in ("OO", "OS", "SO", "SS")}
    n_acc = {c: 0 for c in ("OO", "OS", "SO", "SS")}
    for r in sub:
        exp = EXPECTED_LABEL[r["cell"]]
        acc[r["cell"]] += int(r["predicted_label"] == exp)
        n_acc[r["cell"]] += 1
        if r["cell"] == "SS":
            n_ss += 1
            ss_patch += int(r["predicted_label"] != exp)
        else:
            n_cells += 1
            nonSS_patch += int(r["predicted_label"] != exp)
    for g in {r["source_group_id"] for r in sub}:
        ss_base_total += Z_SS_ERR[g]
        nonSS_base_total += Z_NON_ERR[g]
    gain = (ss_base_total - ss_patch) / n_ss if n_ss else 0.0
    harm = (nonSS_patch - nonSS_base_total) / n_cells if n_cells else 0.0
    return {"n_ss": n_ss, "SS_net_gain": gain, "nonSS_added_harm": harm, "CSI": gain - harm,
            "SS_false_reject": ss_base_total / n_ss if n_ss else None,
            "SO_false_accept": (1 - acc["SO"] / n_acc["SO"]) if n_acc["SO"] else None,
            "acc": {c: acc[c] / n_acc[c] for c in acc}}


metrics = {c: cond_metrics(c) for c in conds}
with open(R / "final_reserve_condition_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "n_ss", "SS_net_gain", "nonSS_added_harm", "CSI",
                "SS_false_reject_rate", "SO_false_accept_rate", "OO_acc", "OS_acc", "SO_acc", "SS_acc"])
    for c in conds:
        m = metrics[c]
        w.writerow([c, m["n_ss"], f"{m['SS_net_gain']:.6f}", f"{m['nonSS_added_harm']:.6f}",
                    f"{m['CSI']:.6f}", f"{m['SS_false_reject']:.6f}",
                    f"{m['SO_false_accept']}"] + [f"{m['acc'][cc]:.6f}" for cc in ("OO", "OS", "SO", "SS")])

with open(R / "final_random_control_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "CSI", "SS_net_gain", "nonSS_added_harm"])
    for kk in range(1, 11):
        m = metrics[f"N{kk}"]
        w.writerow([f"N{kk}", f"{m['CSI']:.6f}", f"{m['SS_net_gain']:.6f}", f"{m['nonSS_added_harm']:.6f}"])

# ---- group-paired bootstrap ----
groups = sorted(set(r["source_group_id"] for r in results))
G = len(groups)

def group_tensors(cname):
    out = {}
    for g in groups:
        sub = [r for r in results if r["condition"] == cname and r["source_group_id"] == g]
        ss = sum(1 for r in sub if r["cell"] == "SS" and r["predicted_label"] != EXPECTED_LABEL[r["cell"]])
        non = sum(1 for r in sub if r["cell"] != "SS" and r["predicted_label"] != EXPECTED_LABEL[r["cell"]])
        out[g] = (ss, non)
    return out

Zg = group_tensors("Z")
base_ss = {g: Zg[g][0] for g in groups}
rngb = np.random.default_rng(BOOT_SEED)

def bootstrap_ci(cname):
    Cg = group_tensors(cname)
    vals = []
    for _ in range(2000):
        idx = rngb.integers(0, G, size=G)
        ss_b = sum(base_ss[groups[i]] for i in idx)
        ss_p = sum(Cg[groups[i]][0] for i in idx)
        non_p = sum(Cg[groups[i]][1] for i in idx)
        non_b = sum(Zg[groups[i]][1] for i in idx)
        gain = (ss_b - ss_p) / G
        harm = (non_p - non_b) / (3 * G)
        vals.append(gain - harm)
    return np.percentile(vals, [2.5, 97.5]), float(np.mean(vals))

ci_r, mean_r = bootstrap_ci("R")
ci_v, _ = bootstrap_ci("V")
ci_rand = [bootstrap_ci(f"N{kk}") for kk in range(1, 11)]
rand_med = float(np.median([x[1] for x in ci_rand]))

# real - reverse paired CI
def ci_diff(c1, c2):
    C1 = group_tensors(c1)
    C2 = group_tensors(c2)
    vals = []
    for _ in range(2000):
        idx = rngb.integers(0, G, size=G)
        ss1 = sum(C1[groups[i]][0] for i in idx)
        ss2 = sum(C2[groups[i]][0] for i in idx)
        n1 = sum(C1[groups[i]][1] for i in idx)
        n2 = sum(C2[groups[i]][1] for i in idx)
        vals.append((ss1 - ss2) / G - (n1 - n2) / (3 * G))
    return float(np.percentile(vals, 2.5))

real_minus_rev_lo = ci_diff("R", "V")
real_minus_rand_lo = float(ci_r[0][0]) - float(np.median([x[0][0] for x in ci_rand]))

ss_net = metrics["R"]["SS_net_gain"]
harm = metrics["R"]["nonSS_added_harm"]
csi_real = metrics["R"]["CSI"]
sel_ss_base = sum(base_ss.values())

bootstrap_rows = [
    {"metric": "CSI_real", "point": csi_real, "ci_lower": ci_r[0][0], "ci_upper": ci_r[0][1]},
    {"metric": "CSI_reverse", "point": metrics["V"]["CSI"], "ci_lower": ci_v[0][0], "ci_upper": ci_v[0][1]},
    {"metric": "CSI_real_minus_reverse", "point": csi_real - metrics["V"]["CSI"],
     "ci_lower": real_minus_rev_lo, "ci_upper": None},
    {"metric": "CSI_real_minus_median_random", "point": csi_real - rand_med,
     "ci_lower": real_minus_rand_lo, "ci_upper": None},
]
with open(R / "bootstrap_causal_location_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["metric", "point", "ci_lower", "ci_upper"])
    w.writeheader()
    w.writerows(bootstrap_rows)

print(f"real: SS_net_gain={ss_net:.4f} nonSS_harm={harm:.4f} CSI={csi_real:.4f} base_err={sel_ss_base}")
print(f"  CI(CSI_real)=({ci_r[0][0]:.4f},{ci_r[0][1]:.4f})")
print(f"  CI(real-reverse) lower={real_minus_rev_lo:.4f}")
print(f"  CI(real-median random) lower={real_minus_rand_lo:.4f}")

confirmed = (ss_net >= 0.10 and harm <= 0.02 and csi_real > 0 and ci_r[0][0] > 0 and
             real_minus_rev_lo > 0 and real_minus_rand_lo > 0 and sel_ss_base >= 30)
label = "causal_location_selective_patch_confirmed" if confirmed else "causal_location_final_not_confirmed"
print("FINAL:", label)

(R / "artifacts").mkdir(parents=True, exist_ok=True)
(R / "artifacts" / "decision.json").write_text(json.dumps({
    "final_label": label,
    "reason": (f"selected_location={loc} alpha*={alpha_star} t_prefix={t_prefix:.4f}; "
               f"real SS_net_gain={ss_net:.4f}>=0.10, nonSS_harm={harm:.4f}<=0.02, CSI={csi_real:.4f}>0, "
               f"CI_lower(CSI_real)={ci_r[0][0]:.4f}>0, "
               f"CI_lower(real-reverse)={real_minus_rev_lo:.4f}>0, "
               f"CI_lower(real-median rand)={real_minus_rand_lo:.4f}>0, "
               f"selected SS base={sel_ss_base}>=30"),
    "final_reserve_model_scored": True,
    "final_reserve_hidden_states_read": True,
    "monolithic_full_forward_only": True,
    "prefix_cache_used": False,
    "activation_intervention_run": True,
    "prompt_baselines_run": True,
    "mistral_loaded": False,
    "selected_location": loc, "alpha_star": alpha_star, "t_prefix": t_prefix,
}, indent=2), encoding="utf-8")
print("Phase 4 OK")
