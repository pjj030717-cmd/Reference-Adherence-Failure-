#!/usr/bin/env python3
"""D3-D Phase 1: train 587 SS inputs, extract 12 location features
((R_end, C_end, D_pos) x (L14, L18, L22, L26)) via monolithic forward,
then per-location direction construction with 5-fold group-stratified OOF.

每 location 独立选择方向（V_mean / V_lda / V_logit C in {0.001,0.01,0.1}），
选择规则固定：max OOF AUPRC；差距<=0.005 则更高 AUROC；并列 V_mean>V_lda>V_logit。
随后用全部 587 重拟合并冻结 v[L,pos], mu, sigma_z, method。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch

import d3d_core as C

D3MR1 = REPO_ROOT / "d3mr1_qwen25_7b_monolithic_prefix_direction_intervention_20260802"
R = REPO_ROOT / "d3d_qwen25_7b_monolithic_causal_location_scan_20260803"

FOLD_SEED = 20260811


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


pairs = C.load_swap_pairs("train")
assert len(pairs) == 587
C.get_model()

# ---- 1.1 extract features + labels ----
# label: SS 错拒 (pred B) = 1, SS 接受 (pred A) = 0, 用完整单体前向
rows = []
for i, p in enumerate(pairs):
    gid = p["original_group_id"]
    s = C.score_monolithic(p["q"], p["r_s"], p["c_s"])
    y = 1 if s["predicted_label"] == "B" else 0
    feats, _ = C.extract_all_positions(p["q"], p["r_s"], p["c_s"], C.CAND_LAYERS)
    rows.append({"source_group_id": gid, "y": y, "d_raw": s["d_raw"],
                 "r_end_pos": s["r_end_pos"], "c_end_pos": s["c_end_pos"], "d_pos": s["d_pos"],
                 "feats": feats})
    if (i + 1) % 50 == 0:
        print(f"  extract {i+1}/587")

loc_names = [f"L{li}/{pos}" for li in C.CAND_LAYERS for pos in C.POSITIONS]

# ---- 1.2 per-location direction selection (5-fold OOF) ----
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(FOLD_SEED)
n = len(rows)
y = np.array([r["y"] for r in rows], dtype=np.int64)
perm = rng.permutation(n)
folds = []
for f in range(5):
    folds.append(set(perm[f::5].tolist()))

ORDER = {"V_mean": 0, "V_lda": 1, "V_logit": 2}


def fit_direction(kind, Cval, x_tr, y_tr):
    """Standardize within fold-training, build direction, map back to raw coords.
    Return unit v_raw (raw coords) with sign mean(z|y=1)>mean(z|y=0)."""
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
    v_raw = v_std / sd
    v_raw = v_raw / (np.linalg.norm(v_raw) + 1e-12)
    z = x_tr @ v_raw - mu @ v_raw
    if z[y_tr == 1].mean() < z[y_tr == 0].mean():
        v_raw = -v_raw
    return v_raw, mu


def z_of(x, v_raw, mu):
    return x @ v_raw - mu @ v_raw


oof_records = []
selection = {}
for loc in loc_names:
    X = np.stack([r["feats"][loc] for r in rows], axis=0).astype(np.float64)
    cand = []
    for kind in ["V_mean", "V_lda", "V_logit"]:
        Cvals = [None] if kind != "V_logit" else [0.001, 0.01, 0.1]
        for cv in Cvals:
            key = kind if cv is None else f"V_logit@C={cv}"
            z_oof = np.zeros(n, dtype=np.float64)
            for f in range(5):
                tr_idx = np.array([i for i in range(n) if i not in folds[f]])
                te_idx = np.array(sorted(folds[f]))
                v_raw, mu = fit_direction(kind, cv, X[tr_idx], y[tr_idx])
                z_oof[te_idx] = z_of(X[te_idx], v_raw, mu)
            auroc = float(roc_auc_score(y, z_oof))
            auprc = float(average_precision_score(y, z_oof))
            cand.append({"method": key, "AUROC": round(auroc, 6), "AUPRC": round(auprc, 6)})
            oof_records.append({"location": loc, "method": key,
                                "AUROC": round(auroc, 6), "AUPRC": round(auprc, 6)})
    cand_sorted = sorted(cand, key=lambda r: (-r["AUPRC"], -r["AUROC"]))
    best = cand_sorted[0]
    # tiebreak within 0.005
    tier = [r for r in cand if abs(r["AUPRC"] - best["AUPRC"]) <= 0.005]
    tier.sort(key=lambda r: (-r["AUROC"], ORDER[r["method"].split("@")[0]]))
    best = tier[0]
    selection[loc] = best
    print(f"  {loc}: selected {best['method']} (AUROC {best['AUROC']}, AUPRC {best['AUPRC']})")

with open(R / "train_location_direction_oof_metrics.csv", "w", newline="") as f:
    import csv
    w = csv.DictWriter(f, fieldnames=["location", "method", "AUROC", "AUPRC"])
    w.writeheader()
    w.writerows(oof_records)

# ---- refit on all train & freeze ----
frozen = {"direction_method_per_location": {}, "fold_seed": FOLD_SEED,
          "locations": loc_names, "n_train": n}
feats_all = {}
for loc in loc_names:
    X = np.stack([r["feats"][loc] for r in rows], axis=0).astype(np.float64)
    kind = selection[loc]["method"].split("@")[0]
    cv = None if kind != "V_logit" else float(selection[loc]["method"].split("=")[1])
    v_raw, mu = fit_direction(kind, cv, X, y)
    sigma_z = float(np.std(z_of(X, v_raw, mu)))
    feats_all[loc] = {"v_raw": v_raw.astype(np.float16), "mu": mu.astype(np.float16),
                      "sigma_z": sigma_z, "z_all": z_of(X, v_raw, mu).astype(np.float16)}
    frozen["direction_method_per_location"][loc] = selection[loc]["method"]
    frozen[f"sigma_z/{loc}"] = sigma_z
    print(f"  freeze {loc}: sigma_z={sigma_z:.4f}")

np.savez(R / "train_location_features.npz",
         **{f"{loc.replace('/', '_')}": np.stack([r["feats"][loc] for r in rows], axis=0).astype(np.float16)
            for loc in loc_names})
np.save(R / "train_ss_labels.npy", y)
json.dump([r["source_group_id"] for r in rows], open(R / "train_ss_gids.json", "w"))
json.dump([r["d_raw"] for r in rows], open(R / "train_ss_draw.json", "w"))

# freeze artifact per location
np.savez(R / "train_location_direction_artifact.npz",
         **{f"{loc.replace('/', '_')}_v": feats_all[loc]["v_raw"] for loc in loc_names},
         **{f"{loc.replace('/', '_')}_mu": feats_all[loc]["mu"] for loc in loc_names},
         **{f"{loc.replace('/', '_')}_sigma": np.array([feats_all[loc]["sigma_z"]]) for loc in loc_names})
json.dump(frozen, open(R / "train_location_direction_selection.json", "w"), indent=1)


y1 = int(y.sum())
y0 = int(n - y1)
print(f"train SS labels: y1={y1} y0={y0} total={n}")
print("Phase 1 OK")
