#!/usr/bin/env python3
"""D3-M Phase 1: full-input intervention direction construction.

Steps:
 1. Deterministic D3M-fit/D3M-tune split of D0 train 587 groups
    (sort by sha256(group_id) ascending; shuffle with random.Random(20260804); 70/30).
 2. Re-score every train SS cell with the full monolithic forward -> labels
    y=1 (wrongly rejected B) / y=0 (correctly accepted A). Cross-check D2-R1.
 3. Extract L18/R_end hidden state (monolithic forward, capture hook) as features.
 4. Capacity gate: fit & tune each y1>=80 and y0>=40.
 5. Fit StandardScaler(fit only) + LogisticRegression(C=0.01, balanced, 20260804).
 6. Direction v = coef/scaler.scale normalized; sign check.
 7. Tune qualification: AUROC >= 0.65 and > 97.5pct of 200 permutation nulls.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import d3m_core as C

D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
M = REPO_ROOT / "d3m_qwen25_7b_monolithic_reference_binding_intervention_20260802"
SEED = 20260804


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (M / "artifacts").mkdir(parents=True, exist_ok=True)
    (M / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
        "monolithic_full_forward_only": True, "segmented_execution_used": False,
        "prefix_cache_used": False, "activation_intervention_run": False,
        "prompt_baselines_run": False, "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


def gid_sha(g):
    return hashlib.sha256(g.encode("utf-8")).hexdigest()


train_pairs = C.load_swap_pairs("train")
gids = [p["original_group_id"] for p in train_pairs]
assert len(gids) == len(set(gids)) == 587
order = sorted(gids, key=gid_sha)
rng = random.Random(SEED)
rng.shuffle(order)
n_fit = int(round(587 * 0.70))
fit_ids = set(order[:n_fit])
tune_ids = set(order[n_fit:])
assert len(fit_ids) == n_fit and len(tune_ids) == 587 - n_fit

# manifest
manifest = {
    "seed": SEED, "method": "sort by sha256(source_group_id) ascending; shuffle random.Random(20260804); 70/30",
    "n_total": 587, "n_fit": n_fit, "n_tune": 587 - n_fit,
    "groups": [
        {"source_group_id": g, "sha256": gid_sha(g), "subset": "fit" if g in fit_ids else "tune"}
        for g in order
    ],
}
(M / "train_tune_split_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
print(f"fit={n_fit} tune={587-n_fit}")

# ---- cross-check labels with D2-R1 train SS scores ----
d2r1_train = {r["source_group_id"]: r for r in json.loads((D2R1 / "scripts" / "_ss_train_scores.json").read_text(encoding="utf-8"))}
assert len(d2r1_train) == 587

# ---- score all train SS + extract L18/R_end feature ----
rows = []
for i, p in enumerate(train_pairs):
    gid = p["original_group_id"]
    # SS cell
    s = C.run_intervention(p["q"], p["r_s"], p["c_s"], apply_fn=None, capture={})
    cap = s["capture"]
    feat = cap["pre"].squeeze(0).float().numpy()  # (3584,)
    y = 1 if s["predicted_label"] == "B" else 0
    d2 = d2r1_train[gid]
    y_cross = 1 if d2["predicted_label"] == "B" else 0
    rows.append({
        "source_group_id": gid, "subset": "fit" if gid in fit_ids else "tune",
        "l_A": s["l_A"], "l_B": s["l_B"], "d_raw": s["d_raw"],
        "predicted_label": s["predicted_label"], "y": y,
        "d2r1_predicted_label": d2["predicted_label"], "y_cross": y_cross,
        "label_agrees_d2r1": bool(y == y_cross),
        "feat": feat,
    })
    if i % 50 == 0:
        print(f"  scored {i}/587")

cross_mismatch = sum(1 for r in rows if not r["label_agrees_d2r1"])
print(f"cross-check vs D2-R1: {587-cross_mismatch}/587 agree")

with open(M / "train_ss_fullforward.json", "w") as f:
    json.dump([{k: v for k, v in r.items() if k != "feat"} for r in rows], f, indent=1)
np.save(M / "train_ss_l18_rend.npy", np.stack([r["feat"] for r in rows], axis=0).astype(np.float16))
with open(M / "train_ss_gids.json", "w") as f:
    json.dump([r["source_group_id"] for r in rows], f)

# ---- capacity ----
for subset in ("fit", "tune"):
    sub = [r for r in rows if r["subset"] == subset]
    y1 = sum(1 for r in sub if r["y"] == 1)
    y0 = sum(1 for r in sub if r["y"] == 0)
    print(f"  {subset}: n={len(sub)} y1={y1} y0={y0}")
    if y1 < 80 or y0 < 40:
        fail("monolithic_direction_label_capacity_insufficient",
             f"{subset} y1={y1} y0={y0} (need >=80 / >=40)")

# ---- fit probe on fit only ----
fit = [r for r in rows if r["subset"] == "fit"]
tune = [r for r in rows if r["subset"] == "tune"]
X_fit = np.stack([r["feat"] for r in fit], axis=0).astype(np.float32)
y_fit = np.array([r["y"] for r in fit])
X_tune = np.stack([r["feat"] for r in tune], axis=0).astype(np.float32)
y_tune = np.array([r["y"] for r in tune])

scaler = StandardScaler().fit(X_fit)
Xs_fit = scaler.transform(X_fit)
clf = LogisticRegression(C=0.01, class_weight="balanced", max_iter=10000, random_state=SEED)
clf.fit(Xs_fit, y_fit)

v_raw = clf.coef_[0] / scaler.scale_
v = v_raw / np.linalg.norm(v_raw)
sigma_proj = np.std(X_fit @ v)  # std(v.h) on D3M-fit only

# sign check: higher v.h should predict y=1
scores_fit = X_fit @ v
corr = np.corrcoef(scores_fit, y_fit)[0, 1]
sign_flipped = corr < 0
if sign_flipped:
    v = -v
    scores_fit = -scores_fit
    print("sign flipped: corr was negative, v *= -1")

# risk scores on tune (probe logit probability)
p_fit = clf.predict_proba(scaler.transform(X_fit))[:, 1]
p_tune = clf.predict_proba(scaler.transform(X_tune))[:, 1]

# ---- tune qualification ----
auroc_tune = roc_auc_score(y_tune, p_tune)
nulls = []
rng2 = np.random.RandomState(SEED)
for _ in range(200):
    yp = rng2.permutation(y_tune)
    nulls.append(roc_auc_score(yp, p_tune))
nulls = np.array(nulls)
p97_5 = np.percentile(nulls, 97.5)
print(f"tune AUROC={auroc_tune:.4f} null97.5={p97_5:.4f}")
if auroc_tune < 0.65 or auroc_tune <= p97_5:
    fail("full_context_direction_signal_insufficient",
         f"AUROC={auroc_tune:.4f} < 0.65 or <= null97.5={p97_5:.4f}")

# ---- save frozen direction artifacts ----
np.savez(M / "frozen_direction.npz",
         v=v.astype(np.float32), v_raw=v_raw.astype(np.float32),
         sigma_proj=np.float32(sigma_proj), auroc_tune=np.float32(auroc_tune))
dir_sha = hashlib.sha256(v.tobytes()).hexdigest()
json.dump({
    "seed": SEED, "C": 0.01, "class_weight": "balanced", "max_iter": 10000,
    "scaler_fit_on": "D3M-fit", "n_fit": len(fit), "n_tune": len(tune),
    "auroc_tune": float(auroc_tune), "null_97_5": float(p97_5),
    "sigma_proj": float(sigma_proj), "sign_flipped": sign_flipped,
    "direction_sha256": dir_sha, "corr_fit": float(corr),
    "capacity_fit_y1_y0": [sum(1 for r in fit if r["y"] == 1), sum(1 for r in fit if r["y"] == 0)],
    "capacity_tune_y1_y0": [sum(1 for r in tune if r["y"] == 1), sum(1 for r in tune if r["y"] == 0)],
    "cross_check_d2r1_agree": 587 - cross_mismatch,
}, open(M / "full_context_direction_fit_audit.json", "w"), indent=2)

# tune metrics csv
with open(M / "full_context_direction_tune_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "subset", "y", "p_error", "v_dot_h"])
    w.writeheader()
    for r, p, s in zip(tune, p_tune, X_tune @ v):
        w.writerow({"source_group_id": r["source_group_id"], "subset": "tune",
                    "y": r["y"], "p_error": float(p), "v_dot_h": float(s)})
print("Phase 1 OK; direction_sha256 =", dir_sha)
