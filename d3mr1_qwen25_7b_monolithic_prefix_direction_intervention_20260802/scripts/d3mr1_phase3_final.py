#!/usr/bin/env python3
"""D3-M-R1 Phase 3: final-reserve one-shot confirmation (first-ever read).

Conditions on groups selected by frozen prefix risk score (frozen q*):
  Z: zero/no patch
  R: real direction  h + alpha*sigma_z*v*
  V: reverse direction h - alpha*sigma_z*v*
  N1..N10: fixed random unit dirs (seed 20260806, N(0,I), removed projection on v*, L2 norm,
           same alpha* and sigma_z)
All: full monolithic forward, selected groups' four cells, batch=1, BF16, eval,
     inference_mode. Record per-cell A/B, d_raw, hook success.
Bootstrap 2000 (seed 20260807). Confirmation gates.
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

D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
R = REPO_ROOT / "d3mr1_qwen25_7b_monolithic_prefix_direction_intervention_20260802"

ALPHAS = [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0]
RANDOM_SEED = 20260806
BOOT_SEED = 20260807


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
art = np.load(R / "frozen_direction_artifact.npz")
v_raw = art["v_raw"].astype(np.float64)
mu_train = art["mu_train"].astype(np.float64)
sigma_z = float(art["sigma_z_train"])
alpha_star = freeze["alpha_star"]
q_star = freeze["q_star"]

# ---- select final-reserve groups by frozen risk score ----
pairs = C.load_swap_pairs("final_reserve")
assert len(pairs) == 197

C.get_model()
sel_scores = []
D2R1_HID = D2R1 / "prefix_hidden_states"
for p in pairs:
    gid = p["original_group_id"]
    h, plen = C.prefix_l18_render(p["q"], p["r_s"])
    z = float((h.astype(np.float64) - mu_train) @ v_raw)
    sel_scores.append({"source_group_id": gid, "z_dev": z, "prefix_len": plen})
sel_scores.sort(key=lambda r: -r["z_dev"])
n_sel = int(round(197 * q_star))
sel_ids = {r["source_group_id"] for r in sel_scores[:n_sel]}
print(f"final-reserve selected: {n_sel}/{197} (q*={q_star})")

# ---- random directions (fixed seed, no projection on v*, L2 norm) ----
rng = np.random.default_rng(RANDOM_SEED)
rand_dirs = []
for k in range(10):
    d = rng.standard_normal(3584)
    d = d - (d @ v_raw) * v_raw
    d = d / (np.linalg.norm(d) + 1e-12)
    rand_dirs.append(d)

# ---- per-cell base error (D1 expected labels) ----
# final-reserve not in D1; we need expected labels from D0 pair definition.
# expected: OO=A correct, OS=B incorrect, SO=B incorrect, SS=A correct
def expected_label(cell):
    return {"OO": "A", "OS": "B", "SO": "B", "SS": "A"}[cell]


def run_condition(p, delta_vec, capture=None):
    """delta_vec: numpy vector to add (scaled by sigma_z already for real/reverse)."""
    if delta_vec is None:
        return C.run_intervention(p["q"], p["r_o"], p["c_o"], apply_fn=None)
    delta_t = torch.tensor(delta_vec, dtype=torch.float32, device="cuda")

    def apply_fn(hidden):
        return hidden.to(torch.float32) + delta_t
    return C.run_intervention(p["q"], p["r_o"], p["c_o"], apply_fn=apply_fn)


# conditions: "Z", "R", "V", "N1".."N10"
conditions = {
    "Z": None,
    "R": alpha_star * sigma_z * v_raw,
    "V": -alpha_star * sigma_z * v_raw,
}
for k, d in enumerate(rand_dirs, 1):
    conditions[f"N{k}"] = alpha_star * sigma_z * d

# ---- run ----
# We score each selected group's four cells per condition. Also need Z baseline for all.
cell_expected = {}
results = []  # list of dicts: source_group_id, cell, condition, predicted_label, d_raw, hook_ok
for p in pairs:
    gid = p["original_group_id"]
    if gid not in sel_ids:
        continue
    for cell, ref, cand, exp in C.four_cells(p):
        for cond, delta in conditions.items():
            if delta is None:
                s = C.run_intervention(p["q"], ref, cand, apply_fn=None)
            else:
                delta_t = torch.tensor(delta, dtype=torch.float32, device="cuda")

                def apply_fn(hidden, _dt=delta_t):
                    return hidden.to(torch.float32) + _dt
                s = C.run_intervention(p["q"], ref, cand, apply_fn=apply_fn)
            results.append({
                "source_group_id": gid, "cell": cell, "condition": cond,
                "predicted_label": s["predicted_label"], "d_raw": s["d_raw"],
                "r_end_pos": s["r_end_pos"], "seq_len": s["seq_len"],
            })
    print(f"  {gid[:12]} done")

with open(R / "final_reserve_group_level_effects.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

# ---- metrics per condition ----
def cond_metrics(results, cond):
    sub = [r for r in results if r["condition"] == cond]
    ss_base = ss_patch = 0
    nonSS_patch = n_cells = n_ss = 0
    acc = {"OO": 0, "OS": 0, "SO": 0, "SS": 0}
    n_acc = {"OO": 0, "OS": 0, "SO": 0, "SS": 0}
    for r in sub:
        exp = expected_label(r["cell"])
        correct = (r["predicted_label"] == exp)
        acc[r["cell"]] += int(correct)
        n_acc[r["cell"]] += 1
        if r["cell"] == "SS":
            n_ss += 1
            if exp == "A" and r["predicted_label"] == "B":
                ss_base += 1  # false reject count (label errors)
            ss_patch += (1 if r["predicted_label"] != exp else 0)
        else:
            n_cells += 1
            nonSS_patch += (1 if r["predicted_label"] != exp else 0)
    return {
        "n_groups": n_ss,
        "SS_net_gain": (ss_base - ss_patch) / n_ss,
        "nonSS_added_harm": nonSS_patch / n_cells,
        "CSI": (ss_base - ss_patch) / n_ss - nonSS_patch / n_cells,
        "SS_false_reject_rate": ss_base / n_ss,
        "SO_false_accept_rate": acc["SO"] / n_acc["SO"] if n_acc["SO"] else None,
        "acc": {c: acc[c] / n_acc[c] for c in acc},
    }


metrics = {c: cond_metrics(results, c) for c in conditions}
with open(R / "final_reserve_condition_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "n_groups", "SS_net_gain", "nonSS_added_harm", "CSI",
                "SS_false_reject_rate", "SO_false_accept_rate", "OO_acc", "OS_acc", "SO_acc", "SS_acc"])
    for c, m in metrics.items():
        w.writerow([c, m["n_groups"], f"{m['SS_net_gain']:.6f}", f"{m['nonSS_added_harm']:.6f}",
                    f"{m['CSI']:.6f}", f"{m['SS_false_reject_rate']:.6f}",
                    f"{m['SO_false_accept_rate']}", *[f"{m['acc'][k]:.6f}" for k in ("OO", "OS", "SO", "SS")]])

# random controls metrics
with open(R / "random_direction_control_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "CSI", "SS_net_gain", "nonSS_added_harm"])
    for c in [f"N{k}" for k in range(1, 11)]:
        m = metrics[c]
        w.writerow([c, f"{m['CSI']:.6f}", f"{m['SS_net_gain']:.6f}", f"{m['nonSS_added_harm']:.6f}"])

# ---- bootstrap (source-group paired, 2000, seed) ----
# data: per selected group, per condition, per cell -> error
# build group-level (or cell-level) bootstrap with group pairing
groups = sorted(set(r["source_group_id"] for r in results))
G = len(groups)
# condition metrics recomputed at group level
def group_tensors(cond):
    """for each group, list of [ss_err_base(=label error at Z), ss_err_patch, nonSS_err_patch]"""
    out = {}
    for g in groups:
        sub = [r for r in results if r["condition"] == cond and r["source_group_id"] == g]
        ss_label_err = 0  # errors in SS (either direction)
        nonSS_err = 0
        for r in sub:
            exp = expected_label(r["cell"])
            if r["cell"] == "SS":
                ss_label_err += int(r["predicted_label"] != exp)
            else:
                nonSS_err += int(r["predicted_label"] != exp)
        out[g] = {"ss": ss_label_err, "non": nonSS_err}
    return out

Zg = group_tensors("Z")
# base SS error (per group) comes from Z condition
base_ss = {g: Zg[g]["ss"] for g in groups}
rngb = np.random.default_rng(BOOT_SEED)

def bootstrap_ci(cond):
    Cg = group_tensors(cond)
    gains = []
    for _ in range(2000):
        idx = rngb.integers(0, G, size=G)
        gsum = lambda f: sum(f(groups[i]) for i in idx)
        ss_base = sum(base_ss[groups[i]] for i in idx)
        ss_patch = gsum(lambda g: Cg[g]["ss"])
        nonSS = gsum(lambda g: Cg[g]["non"])
        gain = (ss_base - ss_patch) / G
        harm = nonSS / (3 * G)
        gains.append(gain - harm)
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(gains))

csi_real = metrics["R"]["CSI"]
ci_real = bootstrap_ci("R")
ci_rev = bootstrap_ci("V")
ci_rands = [bootstrap_ci(f"N{k}") for k in range(1, 11)]
rand_med = np.median([ci_rands[k][2] for k in range(10)])
# real - reverse CI
def ci_diff(c1, c2):
    C1 = group_tensors(c1); C2 = group_tensors(c2)
    out = []
    for _ in range(2000):
        idx = rngb.integers(0, G, size=G)
        s1 = sum(C1[groups[i]]["ss"] for i in idx)
        s2 = sum(C2[groups[i]]["ss"] for i in idx)
        n1 = sum(C1[groups[i]]["non"] for i in idx)
        n2 = sum(C2[groups[i]]["non"] for i in idx)
        c1v = (s1 - s2) / G - (n1 - n2) / (3 * G)
        out.append(c1v)
    return np.percentile(out, 2.5)

real_minus_reverse_lo = ci_diff("R", "V")

# real - median(random) CI
rand_lo = np.median([ci_rands[k][0] for k in range(10)])
diff_real_rand_lo = ci_real[0] - rand_lo  # conservative: real lower bound minus median random lower bound

sel_ss_base_err = sum(base_ss.values())
ss_net = metrics["R"]["SS_net_gain"]
harm = metrics["R"]["nonSS_added_harm"]

bootstrap_rows = [
    {"metric": "CSI_real", "point": csi_real, "ci_lower": ci_real[0], "ci_upper": ci_real[1]},
    {"metric": "CSI_reverse", "point": metrics["V"]["CSI"], "ci_lower": ci_rev[0], "ci_upper": ci_rev[1]},
    {"metric": "CSI_real_minus_reverse", "point": csi_real - metrics["V"]["CSI"],
     "ci_lower": real_minus_reverse_lo, "ci_upper": None},
    {"metric": "CSI_real_minus_median_random", "point": csi_real - rand_med,
     "ci_lower": diff_real_rand_lo, "ci_upper": None},
]
with open(R / "bootstrap_causal_selectivity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["metric", "point", "ci_lower", "ci_upper"])
    w.writeheader()
    w.writerows(bootstrap_rows)

print(f"real SS_net_gain={ss_net:.4f} nonSS_harm={harm:.4f} CSI={csi_real:.4f}")
print(f"  CI real CSI: {ci_real}")
print(f"  CI (real-reverse) lower: {real_minus_reverse_lo:.4f}")
print(f"  CI (real-median random) lower: {diff_real_rand_lo:.4f}")
print(f"  selected SS base errors: {sel_ss_base_err}")

confirmed = (ss_net >= 0.10 and harm <= 0.02 and csi_real > 0 and
             ci_real[0] > 0 and real_minus_reverse_lo > 0 and
             diff_real_rand_lo > 0 and sel_ss_base_err >= 20)
label = "monolithic_patch_selectivity_confirmed" if confirmed else "monolithic_patch_final_not_confirmed"
print("FINAL:", label)

decision = {
    "final_label": label,
    "reason": (f"real SS_net_gain={ss_net:.4f} (>=0.10), nonSS_harm={harm:.4f} (<=0.02), "
               f"CSI={csi_real:.4f} (>0), CI_real>0={ci_real[0]>0}, "
               f"CI(real-reverse)>0={real_minus_reverse_lo>0}, "
               f"CI(real-median random)>0={diff_real_rand_lo>0}, "
               f"selected SS base errors={sel_ss_base_err} (>=20)"),
    "final_reserve_model_scored": True,
    "final_reserve_hidden_states_read": True,
    "monolithic_full_forward_only": True,
    "prefix_cache_used": False,
    "activation_intervention_run": True,
    "prompt_baselines_run": True,
    "mistral_loaded": False,
    "q_star": q_star, "alpha_star": alpha_star,
}
(R / "artifacts").mkdir(parents=True, exist_ok=True)
(R / "artifacts" / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
print("Phase 3 OK")
