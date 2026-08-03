#!/usr/bin/env python3
"""A1 Phase 2: main association analysis.

2A. In-dataset ranking: AUROC / AUPRC + 2,000 bootstrap CIs (resample source groups),
    plus k statistics for y_SS=1 vs y_SS=0, Cliff's delta, Mann-Whitney U p.
2B. Length-controlled logistic: logit P(y_SS=1) = b0 + b1*z(k) + b2*z(q_len) + b3*z(len_r_o) + b4*z(len_r_s)
    within each dataset. z = standardized within the dataset (no y_SS-based transforms).
2C. PopQA relation fixed-effects logistic (16 relations, dummy coded, reference level = first
    in sorted order), plus descriptive relation table (n, SS false-reject rate, median k,
    AUROC(k -> y_SS) only when n>=30 and both classes present).

Logistic fit: hand-rolled IRLS (no regularization). Bootstrap CIs: percentile 2.5-97.5,
    resampling whole source groups with replacement.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = REPO_ROOT / "a1_qwen_no_reference_factual_preference_adherence_audit_20260803"
RNG = np.random.default_rng(20260803)
N_BOOT = 2000
SEED0 = 20260803

df = pd.read_csv(OUT / "factual_preference_scores_dev.csv")
assert list(df.columns[:1]) == ["dataset"]
print("loaded:", len(df), "rows")


def irls_logit(X, y, max_iter=100, tol=1e-9):
    """Binary logistic IRLS, no regularization. X includes intercept column of ones."""
    n, p = X.shape
    beta = np.zeros(p)
    for it in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = mu * (1.0 - mu)
        grad = X.T @ (y - mu)
        # IRLS step: beta_new = beta + (X'WX)^-1 grad
        H = (X * W[:, None]).T @ X
        H += np.eye(p) * 1e-9
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def zscore(x):
    x = np.asarray(x, dtype=float)
    mu, sd = x.mean(), x.std(ddof=0)
    if sd == 0:
        sd = 1.0
    return (x - mu) / sd


# ---------------------------------------------------------------- 2A
def boot_ci_2a(k, y, n_boot=N_BOOT):
    n = len(k)
    aurocs = np.empty(n_boot)
    auprcs = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        kb, yb = k[idx], y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            aurocs[b] = np.nan
            auprcs[b] = np.nan
            continue
        aurocs[b] = roc_auc_score(yb, kb)
        auprcs[b] = average_precision_score(yb, kb)
    return aurocs, auprcs


def cliffs_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n, m = len(x), len(y)
    gt = sum((a > b) for a in x for b in y)
    lt = sum((a < b) for a in x for b in y)
    return (gt - lt) / (n * m)


def summarize_stats(k0, k1):
    def s(v):
        return {"mean": float(np.mean(v)), "median": float(np.median(v)),
                "iqr_lo": float(np.quantile(v, 0.25)), "iqr_hi": float(np.quantile(v, 0.75))}
    return {"ySS0": s(k0), "ySS1": s(k1),
            "cliffs_delta": float(cliffs_delta(k0, k1)),
            "mannwhitney_p": float(stats.mannwhitneyu(k0, k1, alternative="two-sided").pvalue)}


rows2a = []
for ds in ["SciQ", "PopQA"]:
    sub = df[df.dataset == ds]
    k = sub.k.to_numpy()
    y = sub.y_SS.to_numpy()
    k0, k1 = k[y == 0], k[y == 1]
    au, ap = boot_ci_2a(k, y)
    au_valid = au[~np.isnan(au)]
    ap_valid = ap[~np.isnan(ap)]
    rec = {"dataset": ds, "n": int(len(k)), "y_SS1_n": int(y.sum()), "y_SS0_n": int((y == 0).sum()),
           "auroc": float(roc_auc_score(y, k)),
           "auroc_ci_lo": float(np.quantile(au_valid, 0.025)), "auroc_ci_hi": float(np.quantile(au_valid, 0.975)),
           "auprc": float(average_precision_score(y, k)),
           "auprc_ci_lo": float(np.quantile(ap_valid, 0.025)), "auprc_ci_hi": float(np.quantile(ap_valid, 0.975))}
    rec.update(summarize_stats(k0, k1))
    rows2a.append(rec)
    print(ds, "AUROC=%.3f CI[%.3f,%.3f] AUPRC=%.3f Cliff=%.3f p=%.3e" % (
        rec["auroc"], rec["auroc_ci_lo"], rec["auroc_ci_hi"], rec["auprc"],
        rec["cliffs_delta"], rec["mannwhitney_p"]))

with open(OUT / "dataset_level_association_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows2a[0].keys()))
    w.writeheader()
    w.writerows(rows2a)

# ---------------------------------------------------------------- 2B length-controlled logistic
def fit_and_boot_2b(sub, n_boot=N_BOOT):
    k = zscore(sub.k.to_numpy())
    ql = zscore(sub.question_token_length.to_numpy())
    rl_o = zscore(sub.r_o_token_length.to_numpy())
    rl_s = zscore(sub.r_s_token_length.to_numpy())
    y = sub.y_SS.to_numpy()
    X = add_intercept(np.column_stack([k, ql, rl_o, rl_s]))
    beta = irls_logit(X, y)
    # bootstrap by resampling groups (each row is one group here, so resample rows)
    n = len(y)
    b1s = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        bb = irls_logit(Xb, yb)
        b1s[b] = bb[1]
    return beta, b1s


rows2b = []
for ds in ["SciQ", "PopQA"]:
    sub = df[df.dataset == ds]
    beta, b1s = fit_and_boot_2b(sub)
    ci = (np.quantile(b1s, 0.025), np.quantile(b1s, 0.975))
    rows2b.append({"dataset": ds, "n": int(len(sub)),
                   "b1_z(k)": float(beta[1]),
                   "odds_ratio_exp_b1": float(math.exp(beta[1])),
                   "b1_boot_ci_lo": float(ci[0]), "b1_boot_ci_hi": float(ci[1]),
                   "odds_ratio_ci_lo": float(math.exp(ci[0])), "odds_ratio_ci_hi": float(math.exp(ci[1])),
                   "b2_z(q_len)": float(beta[2]), "b3_z(len_r_o)": float(beta[3]), "b4_z(len_r_s)": float(beta[4]),
                   "b0_intercept": float(beta[0])})
    print(ds, "b1=%.3f OR=%.3f CI[%.3f,%.3f]" % (beta[1], math.exp(beta[1]), ci[0], ci[1]))

with open(OUT / "length_controlled_logistic_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows2b[0].keys()))
    w.writeheader()
    w.writerows(rows2b)

# ---------------------------------------------------------------- 2C PopQA relation FE
sub = df[df.dataset == "PopQA"]
rels = sorted(sub.relation.unique())
print("relations:", len(rels), rels)
rel_to_col = {r: i for i, r in enumerate(rels)}
n = len(sub)
k = zscore(sub.k.to_numpy())
ql = zscore(sub.question_token_length.to_numpy())
rl_o = zscore(sub.r_o_token_length.to_numpy())
rl_s = zscore(sub.r_s_token_length.to_numpy())
y = sub.y_SS.to_numpy()
fe = np.zeros((n, len(rels) - 1))
for i, r in enumerate(sub.relation):
    c = rel_to_col[r]
    if c > 0:
        fe[i, c - 1] = 1.0
X = add_intercept(np.column_stack([k, ql, rl_o, rl_s, fe]))
beta = irls_logit(X, y)
# bootstrap (resample groups/rows)
b1s = np.full(N_BOOT, np.nan)
for b in range(N_BOOT):
    idx = RNG.integers(0, n, size=n)
    Xb, yb = X[idx], y[idx]
    if yb.sum() == 0 or yb.sum() == n:
        continue
    bb = irls_logit(Xb, yb)
    b1s[b] = bb[1]
ci = (np.quantile(b1s, 0.025), np.quantile(b1s, 0.975))
fe_coefs = dict(zip(rels[1:], beta[5:]))
rows2c = [{"dataset": "PopQA", "model": "relation_fixed_effects", "n": n,
           "n_relations": len(rels),
           "b1_z(k)": float(beta[1]), "odds_ratio_exp_b1": float(math.exp(beta[1])),
           "b1_boot_ci_lo": float(ci[0]), "b1_boot_ci_hi": float(ci[1]),
           "odds_ratio_ci_lo": float(math.exp(ci[0])), "odds_ratio_ci_hi": float(math.exp(ci[1])),
           "b2_z(q_len)": float(beta[2]), "b3_z(len_r_o)": float(beta[3]), "b4_z(len_r_s)": float(beta[4]),
           "b0_intercept": float(beta[0]),
           "fe_reference_relation": rels[0],
           "fe_coefficients_json": json.dumps(fe_coefs)}]
with open(OUT / "popqa_relation_controlled_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows2c[0].keys()))
    w.writeheader()
    w.writerows(rows2c)
print("PopQA relation-FE b1=%.3f OR=%.3f CI[%.3f,%.3f]" % (beta[1], math.exp(beta[1]), ci[0], ci[1]))

# descriptive relation table
rows_descr = []
for r in rels:
    gr = sub[sub.relation == r]
    nn = len(gr)
    rate = float(gr.y_SS.mean())
    medk = float(gr.k.median())
    both = (gr.y_SS.nunique() == 2)
    auroc = ""
    if nn >= 30 and both:
        try:
            auroc = f"{roc_auc_score(gr.y_SS, gr.k):.3f}"
        except ValueError:
            auroc = "NA"
    rows_descr.append({"relation": r, "n": nn, "ss_false_reject_rate": f"{rate:.4f}",
                       "median_k": f"{medk:.3f}", "auroc_k_to_ySS": auroc,
                       "both_classes": both, "report_auroc": bool(nn >= 30 and both)})
    print(f"  {r}: n={nn} FR_SS={rate:.4f} medk={medk:.3f} auroc={auroc}")
with open(OUT / "popqa_relation_descriptive_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_descr[0].keys()))
    w.writeheader()
    w.writerows(rows_descr)

# ------------------------------------------------- bootstrap_association_audit.csv
# Persist per-iteration bootstrap draws (group-level resampling, seed fixed).
# For each dataset: 2000 draws of AUROC, AUPRC, and length-controlled b1.
# For PopQA additionally: 2000 draws of relation-FE b1.
def boot_2a_draws(k, y):
    nn = len(k)
    au = np.full(N_BOOT, np.nan)
    ap = np.full(N_BOOT, np.nan)
    for b in range(N_BOOT):
        idx = RNG.integers(0, nn, size=nn)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == nn:
            continue
        au[b] = roc_auc_score(yb, k[idx])
        ap[b] = average_precision_score(yb, k[idx])
    return au, ap


boot_rows = []
for ds in ["SciQ", "PopQA"]:
    sub = df[df.dataset == ds]
    k = sub.k.to_numpy(); y = sub.y_SS.to_numpy(); nn = len(k)
    au, ap = boot_2a_draws(k, y)
    kl = zscore(sub.k.to_numpy()); ql = zscore(sub.question_token_length.to_numpy())
    rl_o = zscore(sub.r_o_token_length.to_numpy()); rl_s = zscore(sub.r_s_token_length.to_numpy())
    X = add_intercept(np.column_stack([kl, ql, rl_o, rl_s]))
    b1s = np.full(N_BOOT, np.nan)
    for b in range(N_BOOT):
        idx = RNG.integers(0, nn, size=nn)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == nn:
            continue
        b1s[b] = irls_logit(X[idx], yb)[1]
    for b in range(N_BOOT):
        boot_rows.append({"dataset": ds, "bootstrap_iter": b,
                          "auroc_draw": au[b] if au[b] == au[b] else "NA",
                          "auprc_draw": ap[b] if ap[b] == ap[b] else "NA",
                          "b1_length_controlled_draw": b1s[b] if b1s[b] == b1s[b] else "NA",
                          "b1_relation_fe_draw": "NA"})

# PopQA relation-FE b1 bootstrap draws
sub = df[df.dataset == "PopQA"]
rels = sorted(sub.relation.unique())
rel_to_col = {r: i for i, r in enumerate(rels)}
n = len(sub)
k = zscore(sub.k.to_numpy()); ql = zscore(sub.question_token_length.to_numpy())
rl_o = zscore(sub.r_o_token_length.to_numpy()); rl_s = zscore(sub.r_s_token_length.to_numpy())
y = sub.y_SS.to_numpy()
fe = np.zeros((n, len(rels) - 1))
for i, r in enumerate(sub.relation):
    c = rel_to_col[r]
    if c > 0:
        fe[i, c - 1] = 1.0
X = add_intercept(np.column_stack([k, ql, rl_o, rl_s, fe]))
fe_b1 = np.full(N_BOOT, np.nan)
for b in range(N_BOOT):
    idx = RNG.integers(0, n, size=n)
    yb = y[idx]
    if yb.sum() == 0 or yb.sum() == n:
        continue
    fe_b1[b] = irls_logit(X[idx], yb)[1]
for i, r in enumerate(boot_rows):
    if r["dataset"] == "PopQA":
        b = r["bootstrap_iter"]
        r["b1_relation_fe_draw"] = fe_b1[b] if fe_b1[b] == fe_b1[b] else "NA"

with open(OUT / "bootstrap_association_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "bootstrap_iter", "auroc_draw",
                                      "auprc_draw", "b1_length_controlled_draw",
                                      "b1_relation_fe_draw"])
    w.writeheader()
    w.writerows(boot_rows)
print("bootstrap_association_audit.csv rows:", len(boot_rows))
print("Phase 2 OK")
