#!/usr/bin/env python3
"""A2 Phase 2: frozen comparators + one-shot development evaluation.

Methods:
- B_surface : D2-R1 frozen 9-feature surface baseline (context reference only)
- B_knowledge: k (no-reference factual preference)          [AUROC uses k directly]
- M_rep     : frozen rho (true-prefix representation risk)  [AUROC uses rho directly]
- M_hybrid  : LogisticRegression([z_train(k), z_train(rho)], C=1.0, lbfgs,
              max_iter=1000, random_state=20260819) fitted once on train only;
              dev scored once.

z-transform mean/std computed ONLY from SciQ train (never from dev).
Metrics per method: AUROC, AUPRC, Recall@10% (top 10% by score), group bootstrap
95% CI (2000, seed=20260820, resample source groups).
Core comparisons: dAUROC1 = AUROC(M_rep) - AUROC(B_knowledge);
                  dAUROC2 = AUROC(M_hybrid) - AUROC(B_knowledge);
paired group bootstrap 95% CI for both.
Extra: M_hybrid train coefficients & odds ratios; Spearman corr(k, rho) on dev;
dev k/rho distributions by y.

No hidden states, no PopQA, no final-reserve.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

OUT = REPO_ROOT / "a2_qwen_sciq_rep_increment_over_knowledge_precheck_20260803"
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D4Q1 = REPO_ROOT / "d4q1_qwen25_7b_true_prefix_final_confirmation_20260803"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")

SEED = 20260820
N_BOOT = 2000
RNG = np.random.default_rng(SEED)

k_tr = json.loads((OUT / "scripts" / "_k_train.json").read_text(encoding="utf-8"))
k_de = json.loads((OUT / "scripts" / "_k_dev.json").read_text(encoding="utf-8"))

gids_de = np.array([r["source_group_id"] for r in k_de])
y_de = np.array([r["y"] for r in k_de], dtype=float)
k_de_arr = np.array([r["k"] for r in k_de], dtype=float)
rho_de_arr = np.array([r["rho"] for r in k_de], dtype=float)
k_tr_arr = np.array([r["k"] for r in k_tr], dtype=float)
rho_tr_arr = np.array([r["rho"] for r in k_tr], dtype=float)
y_tr = np.array([r["y"] for r in k_tr], dtype=float)

# ---------------------------------------------------------------------------
# train standardization (train only)
# ---------------------------------------------------------------------------
sc_k = StandardScaler().fit(k_tr_arr.reshape(-1, 1))
sc_rho = StandardScaler().fit(rho_tr_arr.reshape(-1, 1))
z_k_tr = sc_k.transform(k_tr_arr.reshape(-1, 1)).ravel()
z_rho_tr = sc_rho.transform(rho_tr_arr.reshape(-1, 1)).ravel()
z_k_de = sc_k.transform(k_de_arr.reshape(-1, 1)).ravel()
z_rho_de = sc_rho.transform(rho_de_arr.reshape(-1, 1)).ravel()

# ---------------------------------------------------------------------------
# M_hybrid fit on train only (single fit; no CV / no search)
# ---------------------------------------------------------------------------
X_tr = np.column_stack([z_k_tr, z_rho_tr])
hyb = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=20260819)
hyb.fit(X_tr, y_tr)
hyb_score_de = hyb.decision_function(np.column_stack([z_k_de, z_rho_de])).ravel()
print("M_hybrid train coef k/rho:", hyb.coef_[0], "intercept:", hyb.intercept_[0])

# ---------------------------------------------------------------------------
# B_surface dev scores (frozen D2-R1 surface)
# ---------------------------------------------------------------------------
fr = np.load(D4Q1 / "scripts" / "_frozen" / "probe.npz")
tok = AutoTokenizer.from_pretrained(MODEL)
swap_dev = {}
with open(D0 / "preliminary_swap_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d["split"] == "dev":
            swap_dev[d["original_group_id"]] = d
assert len(swap_dev) == 195


def build_surface(gids):
    feats = []
    for g in gids:
        d = swap_dev[g]
        q, r_o, r_s = d["q"], d["r_o"], d["r_s"]
        q_tok = len(tok.encode(q))
        ro_tok = len(tok.encode(r_o))
        rs_tok = len(tok.encode(r_s))
        rs_words = len(r_s.split())
        feats.append([q_tok, ro_tok, rs_tok, abs(ro_tok - rs_tok), len(q), len(r_s),
                      rs_words, 1 if "-" in r_s else 0, 1 if rs_words > 1 else 0])
    return np.array(feats, dtype=float)


Xsur_de = build_surface(gids_de.tolist())
surface_de = ((Xsur_de - fr["surface_scaler_mean"]) / fr["surface_scaler_scale"]) @ fr["surface_coef"].T + fr["surface_intercept"]
surface_de = surface_de.ravel()

# ---------------------------------------------------------------------------
# scores dictionary
# ---------------------------------------------------------------------------
scores = {
    "B_surface": surface_de,
    "B_knowledge": k_de_arr,
    "M_rep": rho_de_arr,
    "M_hybrid": hyb_score_de,
}

n_dev = len(y_de)
pos = int(round(n_dev * 0.10))  # top 10% coverage


def recall_at_coverage(score, y, topk):
    idx = np.argsort(-score)[:topk]
    return float(y[idx].sum() / y.sum())


def auprc(y, s):
    return average_precision_score(y, s)


# ---------------------------------------------------------------------------
# metrics + group bootstrap CI (seed fixed)
# ---------------------------------------------------------------------------
methods = list(scores.keys())
metrics_rows = []
for m in methods:
    s = scores[m]
    au = roc_auc_score(y_de, s)
    ap = average_precision_score(y_de, s)
    rc = recall_at_coverage(s, y_de, pos)
    metrics_rows.append({"method": m, "auroc": au, "auprc": ap, "recall_at_10pct": rc,
                         "n": n_dev, "y1": int(y_de.sum()), "y0": int((1 - y_de).sum()),
                         "coverage_topk": pos})
    print(f"{m:12s} AUROC={au:.4f} AUPRC={ap:.4f} Recall@10%={rc:.4f}")

# bootstrap draws (store per-iteration for audit)
boot = {m: np.full(N_BOOT, np.nan) for m in methods}
for b in range(N_BOOT):
    idx = RNG.integers(0, n_dev, size=n_dev)
    yb = y_de[idx]
    if yb.sum() == 0 or yb.sum() == n_dev:
        continue
    for m in methods:
        sb = scores[m][idx]
        if roc_auc_score(yb, sb) is not None:
            boot[m][b] = roc_auc_score(yb, sb)

for m in methods:
    bv = boot[m][~np.isnan(boot[m])]
    lo, hi = np.quantile(bv, 0.025), np.quantile(bv, 0.975)
    metrics_rows[methods.index(m)]["auroc_ci_lo"] = float(lo)
    metrics_rows[methods.index(m)]["auroc_ci_hi"] = float(hi)
    print(f"  {m} AUROC 95% CI [{lo:.4f}, {hi:.4f}]")

# paired bootstrap for dAUROC1/dAUROC2 (same resample indices)
d1_arr = np.full(N_BOOT, np.nan)
d2_arr = np.full(N_BOOT, np.nan)
for b in range(N_BOOT):
    idx = RNG.integers(0, n_dev, size=n_dev)
    yb = y_de[idx]
    if yb.sum() == 0 or yb.sum() == n_dev:
        continue
    try:
        a_rep = roc_auc_score(yb, scores["M_rep"][idx])
        a_k = roc_auc_score(yb, scores["B_knowledge"][idx])
        a_hyb = roc_auc_score(yb, scores["M_hybrid"][idx])
        d1_arr[b] = a_rep - a_k
        d2_arr[b] = a_hyb - a_k
    except ValueError:
        continue

d1_v = d1_arr[~np.isnan(d1_arr)]
d2_v = d2_arr[~np.isnan(d2_arr)]
d1_ci = (np.quantile(d1_v, 0.025), np.quantile(d1_v, 0.975))
d2_ci = (np.quantile(d2_v, 0.025), np.quantile(d2_v, 0.975))
au_rep, au_k, au_hyb = roc_auc_score(y_de, scores["M_rep"]), roc_auc_score(y_de, scores["B_knowledge"]), roc_auc_score(y_de, scores["M_hybrid"])
print(f"dAUROC1 = {au_rep - au_k:.4f} 95% CI [{d1_ci[0]:.4f}, {d1_ci[1]:.4f}]")
print(f"dAUROC2 = {au_hyb - au_k:.4f} 95% CI [{d2_ci[0]:.4f}, {d2_ci[1]:.4f}]")

# ---------------------------------------------------------------------------
# write outputs
# ---------------------------------------------------------------------------
with open(OUT / "dev_metrics_by_method.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
    w.writeheader()
    w.writerows(metrics_rows)

with open(OUT / "dev_prediction_comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["source_group_id", "y", "score_B_surface", "score_B_knowledge", "score_M_rep", "score_M_hybrid"])
    for i, g in enumerate(gids_de):
        w.writerow([g, int(y_de[i]), scores["B_surface"][i], scores["B_knowledge"][i],
                    scores["M_rep"][i], scores["M_hybrid"][i]])

# paired bootstrap audit csv
with open(OUT / "paired_bootstrap_increment_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["bootstrap_iter", "dAUROC1_Mrep_minus_Bknowledge", "dAUROC2_Mhybrid_minus_Bknowledge"])
    for b in range(N_BOOT):
        w.writerow([b, "" if d1_arr[b] != d1_arr[b] else f"{d1_arr[b]:.6f}",
                    "" if d2_arr[b] != d2_arr[b] else f"{d2_arr[b]:.6f}"])

# train_fit_audit.csv
coef_k, coef_rho = float(hyb.coef_[0][0]), float(hyb.coef_[0][1])
with open(OUT / "train_fit_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["param", "value"])
    w.writeheader()
    for k, v in [("n_train", len(k_tr)), ("y1_train", int(y_tr.sum())), ("y0_train", int((1 - y_tr).sum())),
                 ("coef_z_k", coef_k), ("coef_z_rho", coef_rho),
                 ("odds_ratio_k", math.exp(coef_k)), ("odds_ratio_rho", math.exp(coef_rho)),
                 ("intercept", float(hyb.intercept_[0])),
                 ("C", 1.0), ("solver", "lbfgs"), ("max_iter", 1000), ("random_state", 20260819),
                 ("z_k_train_mean", float(sc_k.mean_[0])), ("z_k_train_std", float(sc_k.scale_[0])),
                 ("z_rho_train_mean", float(sc_rho.mean_[0])), ("z_rho_train_std", float(sc_rho.scale_[0]))]:
        w.writerow({"param": k, "value": v})

# score_relationship_audit.csv
rho_ok = rho_de_arr[(rho_de_arr == rho_de_arr)]
k_ok = k_de_arr[(k_de_arr == k_de_arr)]
spear = spearmanr(k_de_arr, rho_de_arr)
rows_rel = []
rows_rel.append({"metric": "spearman_k_rho_dev", "value": float(spear.statistic)})
rows_rel.append({"metric": "spearman_p", "value": float(spear.pvalue)})
for yv, tag in [(0, "y=0"), (1, "y=1")]:
    kk = k_de_arr[y_de == yv]
    rr = rho_de_arr[y_de == yv]
    rows_rel.append({"metric": f"k_mean_{tag}", "value": float(np.mean(kk))})
    rows_rel.append({"metric": f"k_median_{tag}", "value": float(np.median(kk))})
    rows_rel.append({"metric": f"k_iqr_{tag}", "value": float(np.quantile(kk, 0.75) - np.quantile(kk, 0.25))})
    rows_rel.append({"metric": f"rho_mean_{tag}", "value": float(np.mean(rr))})
    rows_rel.append({"metric": f"rho_median_{tag}", "value": float(np.median(rr))})
    rows_rel.append({"metric": f"rho_iqr_{tag}", "value": float(np.quantile(rr, 0.75) - np.quantile(rr, 0.25))})
with open(OUT / "score_relationship_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["metric", "value"])
    w.writeheader()
    w.writerows(rows_rel)

# train_standardization_and_hybrid_spec.json
json.dump({
    "z_standardization": {
        "k": {"train_mean": float(sc_k.mean_[0]), "train_std": float(sc_k.scale_[0])},
        "rho": {"train_mean": float(sc_rho.mean_[0]), "train_std": float(sc_rho.scale_[0])},
        "note": "mean/std computed ONLY from SciQ train (587 groups); dev transformed with same values."},
    "hybrid": {
        "features": ["z_train(k)", "z_train(rho)"],
        "estimator": "LogisticRegression", "penalty": "l2", "C": 1.0,
        "solver": "lbfgs", "max_iter": 1000, "random_state": 20260819,
        "fitted_once_on": "SciQ train only",
        "no_cv": True, "no_grid_search": True, "no_threshold_optimization": True,
        "dev_scored_once": True,
        "train_coef_k": coef_k, "train_coef_rho": coef_rho,
        "train_odds_ratio_k": math.exp(coef_k), "train_odds_ratio_rho": math.exp(coef_rho),
        "train_intercept": float(hyb.intercept_[0])},
    "pairwise_delta": {
        "dAUROC1_Mrep_minus_Bknowledge": float(au_rep - au_k),
        "dAUROC1_ci95": [float(d1_ci[0]), float(d1_ci[1])],
        "dAUROC2_Mhybrid_minus_Bknowledge": float(au_hyb - au_k),
        "dAUROC2_ci95": [float(d2_ci[0]), float(d2_ci[1])],
    },
}, open(OUT / "train_standardization_and_hybrid_spec.json", "w"), indent=2)

print("Phase 2 OK")
