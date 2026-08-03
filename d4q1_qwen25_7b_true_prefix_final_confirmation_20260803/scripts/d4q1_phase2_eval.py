#!/usr/bin/env python3
"""D4-Q1 Phase 2: one-shot final-reserve H2 evaluation.

Data (all numeric; no raw text):
  - labels y from final_ss_score_and_label_manifest.csv (SS A/B readout; y=1 reject)
  - M_rep scores: frozen D2-R1 Probe (layer18, C=0.01) applied to final prefix h18
  - B_surface scores: frozen surface logistic applied to final surface features

Metrics:
  - AUROC(M_rep), AUPRC(M_rep), AUROC(B_surface), AUPRC(B_surface), delta AUROC/AUPRC
  - capacity gate: n_y1>=30, n_y0>=30
  - group-paired bootstrap (2000, seed 20260812) -> 95% CI of delta AUROC
  - permutation null (200, seed 20260813) on y labels -> 97.5 percentile

Outputs:
  - final_prediction_manifest.csv
  - metrics_final.csv
  - bootstrap_final_metrics.csv
  - permutation_null_final.csv
  - failure_examples.md (hashes + numbers only)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

R = REPO_ROOT / "d4q1_qwen25_7b_true_prefix_final_confirmation_20260803"
LAYER = 18

def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": label, "reason": why,
         "allowed_final_groups": 196, "quarantined_final_groups": 1,
         "quarantined_group_scored": False, "quarantined_group_hidden_state_read": False,
         "final_configuration_changed": False, "hidden_layer": LAYER, "hidden_token": "R_end",
         "probe_C": 0.01, "probe_refit_used_dev": False, "probe_refit_used_final": False,
         "activation_intervention_run": False, "mistral_loaded": False,
         "prompt_baselines_run": False}, indent=2), encoding="utf-8")
    sys.exit(1)

# ---------------------------------------------------------------------------
# load frozen models
# ---------------------------------------------------------------------------
fr = np.load(R / "scripts" / "_frozen" / "probe.npz")
sc_mean, sc_scale = fr["scaler_mean"], fr["scaler_scale"]
coef, intercept = fr["coef"], fr["intercept"]
sc_s_mean, sc_s_scale = fr["surface_scaler_mean"], fr["surface_scaler_scale"]
coef_s, intercept_s = fr["surface_coef"], fr["surface_intercept"]

# ---------------------------------------------------------------------------
# load final data
# ---------------------------------------------------------------------------
labels = {}
with open(R / "final_ss_score_and_label_manifest.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        labels[r["source_group_id"]] = {"l_A": float(r["l_A"]), "l_B": float(r["l_B"]),
                                        "d_raw": float(r["d_raw"]), "y": int(r["y"])}
assert len(labels) == 196, f"labels {len(labels)} != 196"

sf = np.load(R / "scripts" / "_final_surface_feats.npz", allow_pickle=True)
surf_X = sf["X"].astype(float)          # (196, 9)
surf_gids = [str(g) for g in sf["gids"]]

gids = sorted(labels.keys())
assert len(gids) == 196
h18 = np.stack([np.load(R / "prefix_hidden_states" / f"final_{g}.npz")["h_prefix"].astype(np.float32)
                for g in gids])  # (196, 3584)
y = np.array([labels[g]["y"] for g in gids])

# surface features aligned by gid
surf_idx = {g: i for i, g in enumerate(surf_gids)}
X_s = np.stack([surf_X[surf_idx[g]] for g in gids])

# ---------------------------------------------------------------------------
# frozen predictions
# ---------------------------------------------------------------------------
z = (h18 - sc_mean) / sc_scale
s_rep = (z @ coef.T).ravel() + intercept            # M_rep scores
zs = (X_s - sc_s_mean) / sc_s_scale
s_sur = (zs @ coef_s.T).ravel() + intercept_s       # B_surface scores

print("scores finite:", np.isfinite(s_rep).all(), np.isfinite(s_sur).all())
if not (np.isfinite(s_rep).all() and np.isfinite(s_sur).all() and np.isfinite(y).all()):
    fail("inheritance_or_execution_invalid", "NaN/inf in scores or labels")

# unique predictions
pred_rep = (s_rep > 0).astype(int)
pred_sur = (s_sur > 0).astype(int)
uniq_rep = len(set(pred_rep.tolist())) >= 2
print("unique predictions M_rep:", len(set(pred_rep.tolist())), "B_surface:", len(set(pred_sur.tolist())))

# ---------------------------------------------------------------------------
# capacity gate (2.3)
# ---------------------------------------------------------------------------
n_total = len(y)
n_y1 = int(y.sum())
n_y0 = int(n_total - n_y1)
prevalence = n_y1 / n_total
print(f"capacity: n_total={n_total} n_y1={n_y1} n_y0={n_y0} prevalence={prevalence:.4f}")
if n_y1 < 30 or n_y0 < 30:
    fail("final_label_capacity_insufficient", f"n_y1={n_y1} n_y0={n_y0}")

# ---------------------------------------------------------------------------
# metrics (2.4)
# ---------------------------------------------------------------------------
auroc_rep = roc_auc_score(y, s_rep)
auprc_rep = average_precision_score(y, s_rep)
auroc_sur = roc_auc_score(y, s_sur)
auprc_sur = average_precision_score(y, s_sur)
d_auroc = auroc_rep - auroc_sur
d_auprc = auprc_rep - auprc_sur
print(f"AUROC M_rep={auroc_rep:.6f} B_surface={auroc_sur:.6f} dAUROC={d_auroc:+.6f}")
print(f"AUPRC M_rep={auprc_rep:.6f} B_surface={auprc_sur:.6f} dAUPRC={d_auprc:+.6f}")

# ---------------------------------------------------------------------------
# group-paired bootstrap (2000, seed 20260812)
# ---------------------------------------------------------------------------
n_boot = 2000
rng_b = np.random.default_rng(20260812)
boot_d = []
for _ in range(n_boot):
    idx = rng_b.integers(0, n_total, n_total)
    yb, sr, ss = y[idx], s_rep[idx], s_sur[idx]
    if len(np.unique(yb)) < 2:
        boot_d.append(np.nan)
        continue
    a = roc_auc_score(yb, sr)
    b = roc_auc_score(yb, ss)
    boot_d.append(a - b)
boot_d = np.array(boot_d)
lo, hi = np.nanpercentile(boot_d, [2.5, 97.5])
ci = (float(lo), float(hi))
print(f"bootstrap dAUROC CI95 = [{lo:.6f}, {hi:.6f}] lower>0: {lo > 0}")

# also bootstrap each AUROC
boot_rep, boot_sur = [], []
for _ in range(n_boot):
    idx = rng_b.integers(0, n_total, n_total)
    yb, sr, ss = y[idx], s_rep[idx], s_sur[idx]
    if len(np.unique(yb)) < 2:
        boot_rep.append(np.nan); boot_sur.append(np.nan); continue
    boot_rep.append(roc_auc_score(yb, sr)); boot_sur.append(roc_auc_score(yb, ss))
ci_rep = np.nanpercentile(boot_rep, [2.5, 97.5])
ci_sur = np.nanpercentile(boot_sur, [2.5, 97.5])

# ---------------------------------------------------------------------------
# permutation null (200, seed 20260813); permute y only
# ---------------------------------------------------------------------------
n_perm = 200
rng_p = np.random.default_rng(20260813)
null = []
for _ in range(n_perm):
    yp = y[rng_p.permutation(n_total)]
    null.append(roc_auc_score(yp, s_rep))
null = np.array(null)
p975 = float(np.percentile(null, 97.5))
print(f"permutation null: real={auroc_rep:.6f} p97.5={p975:.6f} sig: {auroc_rep > p975}")

# ---------------------------------------------------------------------------
# write outputs
# ---------------------------------------------------------------------------
with open(R / "final_prediction_manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["source_group_id", "y", "score_M_rep", "pred_M_rep", "score_B_surface", "pred_B_surface"])
    for i, g in enumerate(gids):
        w.writerow([g, y[i], s_rep[i], pred_rep[i], s_sur[i], pred_sur[i]])

with open(R / "metrics_final.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerows([
        ["n_total", n_total], ["n_y1", n_y1], ["n_y0", n_y0], ["positive_prevalence", f"{prevalence:.6f}"],
        ["AUROC_M_rep", f"{auroc_rep:.9f}"], ["AUPRC_M_rep", f"{auprc_rep:.9f}"],
        ["AUROC_B_surface", f"{auroc_sur:.9f}"], ["AUPRC_B_surface", f"{auprc_sur:.9f}"],
        ["Delta_AUROC_M_minus_B", f"{d_auroc:.9f}"], ["Delta_AUPRC_M_minus_B", f"{d_auprc:.9f}"],
        ["CI95_AUROC_M_rep", f"[{ci_rep[0]:.9f},{ci_rep[1]:.9f}]"],
        ["CI95_AUROC_B_surface", f"[{ci_sur[0]:.9f},{ci_sur[1]:.9f}]"],
        ["CI95_Delta_AUROC", f"[{ci[0]:.9f},{ci[1]:.9f}]"],
        ["permutation_null_p97_5", f"{p975:.9f}"],
        ["permutation_significant", str(auroc_rep > p975)],
        ["unique_prediction_count", len(set(pred_rep.tolist()))],
    ])

with open(R / "bootstrap_final_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["n_boot", "seed", "metric", "ci_lower_95", "ci_upper_95"])
    w.writerow([n_boot, 20260812, "Delta_AUROC_M_minus_B", f"{ci[0]:.9f}", f"{ci[1]:.9f}"])
    w.writerow([n_boot, 20260812, "AUROC_M_rep", f"{ci_rep[0]:.9f}", f"{ci_rep[1]:.9f}"])
    w.writerow([n_boot, 20260812, "AUROC_B_surface", f"{ci_sur[0]:.9f}", f"{ci_sur[1]:.9f}"])

with open(R / "permutation_null_final.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["n_perm", "seed", "real_AUROC_M_rep", "null_p97_5", "significant"])
    w.writerow([n_perm, 20260813, f"{auroc_rep:.9f}", f"{p975:.9f}", str(auroc_rep > p975)])
    for k, v in enumerate(null):
        w.writerow([f"perm_{k}", "", "", f"{v:.9f}", ""])

# ---------------------------------------------------------------------------
# failure_examples.md (hashes + numbers only)
# ---------------------------------------------------------------------------
order = np.argsort(-(s_rep - s_sur))  # M_rep most confident vs surface
with open(R / "failure_examples.md", "w", encoding="utf-8") as f:
    f.write("# failure_examples.md\n\nD4-Q1 final-reserve：M_rep 相对 B_surface 最自信的预测（仅 hash 与数值，不含题目/答案正文）。\n\n")
    f.write("| group hash | y | d_raw | score_M_rep | pred_M_rep | score_B_surface | pred_B_surface |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for i in order[:20]:
        g = gids[i]
        f.write(f"| {g} | {y[i]} | {labels[g]['d_raw']:.6f} | {s_rep[i]:.6f} | {pred_rep[i]} | "
                f"{s_sur[i]:.6f} | {pred_sur[i]} |\n")

print("Phase 2 outputs written")
# cache for Phase 3
json.dump({"auroc_rep": auroc_rep, "auprc_rep": auprc_rep, "auroc_sur": auroc_sur,
           "auprc_sur": auprc_sur, "d_auroc": d_auroc, "d_auprc": d_auprc,
           "ci_d_auroc": ci, "p975": p975, "n_y1": n_y1, "n_y0": n_y0,
           "n_total": n_total, "unique_preds": len(set(pred_rep.tolist()))},
          open(R / "scripts" / "_phase2_metrics.json", "w"), indent=2)
print("Phase 2 OK")
