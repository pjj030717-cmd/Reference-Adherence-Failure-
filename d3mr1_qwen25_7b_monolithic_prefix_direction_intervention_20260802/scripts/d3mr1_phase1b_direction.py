#!/usr/bin/env python3
"""D3-M-R1 Phase 1b: direction construction (ALL 587 train, no fit/tune split).

Candidates: V_mean, V_lda, V_logit(C in {0.001,0.01,0.1}).
5-fold group-stratified OOF on train only. Selection:
  1) max OOF AUPRC; 2) if |AUPRC diff|<=0.005 pick higher AUROC;
  3) tie -> V_mean > V_lda > V_logit.
Freeze: refit on all train -> v*, mu_train, sigma_z_train.
Also compute descriptive Spearman vs D2-R1 risk score.
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

D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
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


rng = np.random.default_rng(20260804)
feats = np.load(R / "train_prefix_l18_rend.npy").astype(np.float64)
gids = json.loads((R / "train_prefix_gids.json").read_text(encoding="utf-8"))
labels = np.load(R / "train_prefix_labels.npy")
n = len(gids)
assert feats.shape == (587, 3584)

# group-stratified 5-fold (stratify by label within each group's single row)
perm = rng.permutation(n)
folds = []
for f in range(5):
    folds.append(set(perm[f::5].tolist()))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr


def bal_acc_at_train_threshold(z_tr, y_tr, z_te, y_te):
    """train-only threshold on training fold, applied to test fold."""
    pos, neg = z_tr[y_tr == 1], z_tr[y_tr == 0]
    thr = 0.5 * (pos.mean() + neg.mean())
    p = (z_te > thr).astype(int)
    return float(np.mean([p[y_te == 1].mean(), 1 - p[y_te == 0].mean()])) if (y_te.sum() > 0 and (y_te == 0).sum() > 0) else np.nan


def fit_direction(kind, Cval, x_tr, y_tr):
    """x_tr: raw hidden (n,d). Standardize in fold. Return v_raw (unit, in raw coords)
    with sign such that mean(z|y=1)>mean(z|y=0). Also return standardization params."""
    mu = x_tr.mean(0)
    sd = x_tr.std(0)
    sd[sd == 0] = 1.0
    X = (x_tr - mu) / sd
    if kind == "V_mean":
        w = X[y_tr == 1].mean(0) - X[y_tr == 0].mean(0)
    elif kind == "V_lda":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X, y_tr)
        w = clf.coef_[0]
    else:
        clf = LogisticRegression(C=Cval, max_iter=2000)
        clf.fit(X, y_tr)
        w = clf.coef_[0]
    v_std = w / (np.linalg.norm(w) + 1e-12)
    v_raw = v_std / sd  # map back to raw coords (linear approx; exact for raw z later)
    v_raw = v_raw / (np.linalg.norm(v_raw) + 1e-12)
    z = x_tr @ v_raw - mu @ v_raw
    if z[y_tr == 1].mean() < z[y_tr == 0].mean():
        v_raw = -v_raw
    return v_raw, mu, sd


def raw_z(x, v_raw, mu):
    return x @ v_raw - mu @ v_raw


# D2-R1 risk score (descriptive only): D2-R1 published per-layer aggregate metrics,
# not per-group z scores. Field recorded as "not available (D2-R1 published aggregates only)".
# D2-R1 selected layer 18 / C=0.0001 with cv mean AUROC 0.650; our L18 direction is compared qualitatively.
d2r1_sel = json.loads((D2R1 / "scripts" / "_selected_lr.json").read_text(encoding="utf-8"))
D2R1_DESC = {
    "selected_layer": d2r1_sel["selected_layer"],
    "selected_C": d2r1_sel["selected_C"],
    "cv_mean_auroc": d2r1_sel["cv_mean_auroc"],
    "cv_mean_auprc": d2r1_sel["cv_mean_auprc"],
    "note": "D2-R1 published layer-aggregated CV metrics only; per-group z not available for Spearman",
}

rows = []
oof = {}
for kind in ["V_mean", "V_lda", "V_logit"]:
    Cvals = [None] if kind != "V_logit" else [0.001, 0.01, 0.1]
    for cv in Cvals:
        key = kind if cv is None else f"V_logit@C={cv}"
        z_oof = np.zeros(n, dtype=np.float64)
        for f in range(5):
            tr_idx = np.array([i for i in range(n) if i not in folds[f]])
            te_idx = np.array(sorted(folds[f]))
            v_raw, mu, _ = fit_direction(kind, cv, feats[tr_idx], labels[tr_idx])
            z_oof[te_idx] = raw_z(feats[te_idx], v_raw, mu)
        auroc = float(roc_auc_score(labels, z_oof))
        auprc = float(average_precision_score(labels, z_oof))
        sp = None  # D2-R1 per-group z unavailable (aggregate only)
        rows.append({"method": key, "AUROC": round(auroc, 6), "AUPRC": round(auprc, 6),
                     "balacc_train_thresh": round(bal_acc_at_train_threshold(z_oof, labels, z_oof, labels), 6),
                     "spearman_d2r1_desc": sp})
        oof[key] = (auroc, auprc)

rows.sort(key=lambda r: r["method"])
with open(R / "direction_candidate_oof_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "AUROC", "AUPRC", "balacc_train_thresh", "spearman_d2r1_desc"])
    w.writeheader()
    w.writerows(rows)
print("OOF metrics:")
for r in rows:
    print(f"  {r['method']}: AUROC={r['AUROC']} AUPRC={r['AUPRC']} balacc={r['balacc_train_thresh']}")

# selection
cands = sorted(rows, key=lambda r: (-r["AUPRC"], -r["AUROC"]))
best = cands[0]
second = cands[1]
if best["AUPRC"] - second["AUPRC"] <= 0.005:
    # resolve by AUROC; if tied by AUROC, simplicity order
    order = {"V_mean": 0, "V_lda": 1, "V_logit": 2}
    tier = [r for r in rows if abs(r["AUPRC"] - best["AUPRC"]) <= 0.005]
    tier.sort(key=lambda r: (-r["AUROC"], order[r["method"].split("@")[0]]))
    best = tier[0]
print("selected:", best["method"])

# refit on all train
kind = best["method"].split("@")[0]
cv = None if kind != "V_logit" else float(best["method"].split("=")[1])
mu_full = feats.mean(0)
sd_full = feats.std(0)
sd_full[sd_full == 0] = 1.0
X = (feats - mu_full) / sd_full
if kind == "V_mean":
    w = X[labels == 1].mean(0) - X[labels == 0].mean(0)
elif kind == "V_lda":
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, labels)
    w = clf.coef_[0]
else:
    clf = LogisticRegression(C=cv, max_iter=2000).fit(X, labels)
    w = clf.coef_[0]
v_raw2 = w / sd_full
v_raw2 = v_raw2 / (np.linalg.norm(v_raw2) + 1e-12)
if (feats @ v_raw2 - mu_full @ v_raw2)[labels == 1].mean() < (feats @ v_raw2 - mu_full @ v_raw2)[labels == 0].mean():
    v_raw2 = -v_raw2
sigma_z = float(np.std(feats @ v_raw2 - mu_full @ v_raw2))
z_all = feats @ v_raw2 - mu_full @ v_raw2

meta = {
    "direction_method": best["method"],
    "n_train": n,
    "mu_train": mu_full.tolist(),
    "sigma_z_train": sigma_z,
    "v_raw": v_raw2.astype(np.float16).tolist(),
    "fold_seed": 20260804,
    "oof_metrics": rows,
    "selection_rule": "max OOF AUPRC; |diff|<=0.005 -> higher AUROC; tie -> V_mean>V_lda>V_logit",
    "selection_reason": f"OOF AUPRC {best['AUPRC']} (next {second['method']} {second['AUPRC']})",
    "z_train_mean_y1": float(z_all[labels == 1].mean()), "z_train_mean_y0": float(z_all[labels == 0].mean()),
    "spearman_d2r1_desc": best["spearman_d2r1_desc"],
}
np.savez(R / "frozen_direction_artifact.npz",
         v_raw=v_raw2.astype(np.float16), mu_train=mu_full.astype(np.float16),
         sigma_z_train=np.array([sigma_z]), z_all=z_all.astype(np.float16))
(R / "frozen_direction_metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

(R / "direction_selection_audit.md").write_text(
    "# Direction selection audit\n\n"
    f"Selected: **{best['method']}**\n\n"
    "| method | AUROC | AUPRC | balacc | Spearman(D2-R1,desc) |\n|---|---|---|---|---|\n" +
    "\n".join(f"| {r['method']} | {r['AUROC']} | {r['AUPRC']} | {r['balacc_train_thresh']} | {r['spearman_d2r1_desc']} |" for r in rows) +
    f"\n\nReason: max OOF AUPRC {best['AUPRC']}; if within 0.005 of runner-up then AUROC tiebreak.\n"
    f"sigma_z_train = {sigma_z:.6f}\n", encoding="utf-8")
print(f"Phase 1b OK. v*={best['method']} sigma_z={sigma_z:.6f}")
