#!/usr/bin/env python3
"""E01-D2: main analysis.
- ss_label_capacity_audit.csv
- train_cv_by_layer.csv (M_ref_rep 5-fold group CV, layer & C selection)
- metrics_primary_dev.csv (frozen M_ref_rep on dev + 95% group bootstrap CI)
- surface_baseline_spec.json / surface_baseline_metrics.csv
- decision_score_diagnostic.csv
- four_cell_representation_contrasts.csv
- permutation_null_audit.csv (200 permutations)
- swap_overlap_disclosure.md
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
from scipy.stats import sem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
N_LAYERS = 28
HID_DIM = 3584
CS = [0.0001, 0.001, 0.01, 0.1, 1.0]
RNG = np.random.default_rng(20260802)

dev_rows = json.loads((D2 / "scripts" / "_dev_rows.json").read_text(encoding="utf-8"))
train_rows = json.loads((D2 / "scripts" / "_train_rows.json").read_text(encoding="utf-8"))


def ss_y(rows):
    ss = [r for r in rows if r["cell"] == "SS"]
    y = np.array([1 if r["predicted_label"] == "B" else 0 for r in ss])
    return ss, y


dev_ss, y_dev = ss_y(dev_rows)
train_ss, y_train = ss_y(train_rows)
print("train SS n:", len(y_train), "y=1:", y_train.sum(), "y=0:", (1 - y_train).sum())
print("dev   SS n:", len(y_dev), "y=1:", y_dev.sum(), "y=0:", (1 - y_dev).sum())

# ---- capacity audit ----
cap = [["split", "n_ss", "n_error(y=1)", "n_correct(y=0)", "gate"]]
cap.append(["train", len(y_train), int(y_train.sum()), int((1 - y_train).sum()),
            "PASS" if y_train.sum() >= 100 and (1 - y_train).sum() >= 100 else "FAIL"])
cap.append(["dev", len(y_dev), int(y_dev.sum()), int((1 - y_dev).sum()),
            "PASS" if y_dev.sum() >= 30 and (1 - y_dev).sum() >= 30 else "FAIL"])
with open(D2 / "ss_label_capacity_audit.csv", "w", newline="") as f:
    csv.writer(f).writerows(cap)
if cap[1][4] != "PASS" or cap[2][4] != "PASS":
    print("STOP: ss_outcome_capacity_insufficient")
    (D2 / "artifacts").mkdir(parents=True, exist_ok=True)
    (D2 / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": "ss_outcome_capacity_insufficient",
         "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
         "probe_trained": True, "activation_intervention_run": False,
         "prompt_baselines_run": False, "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)
print("capacity gate PASS")


def load_hidden(split_rows):
    """Return dict gid -> {'h_r': (n,28,d) stacked over SS-cell rows} etc for all cells."""
    # we only need SS cell h_r for main probe; all cells for contrasts
    ss_gids = [r["source_group_id"] for r in split_rows if r["cell"] == "SS"]
    cells = ["OO", "OS", "SO", "SS"]
    out = {}
    for gid in ss_gids:
        z = np.load(D2 / "hidden_states" / f"{'dev' if gid in {x['source_group_id'] for x in dev_ss} else 'train'}_{gid}.npz")
        out[gid] = {c: {p: z[f"{c}_h_{p}"] for p in ("r", "c", "d")} for c in cells}
    return out


train_hid = load_hidden(train_ss)
dev_hid = load_hidden(dev_ss)

# feature matrix: SS-cell R_end per layer
def build_X(hid, gids, pos="r"):
    """X[l] shape (n, HID_DIM) for layer l (1..28) at R_end."""
    return np.stack([hid[g]["SS"][pos] for g in gids])  # (n, 28, d)


Xr_tr = np.stack([train_hid[g]["SS"]["r"] for g in [r["source_group_id"] for r in train_ss]])  # (n,28,d)
Xr_dev = np.stack([dev_hid[g]["SS"]["r"] for g in [r["source_group_id"] for r in dev_ss]])
tr_gids = np.array([r["source_group_id"] for r in train_ss])
de_gids = np.array([r["source_group_id"] for r in dev_ss])


def group_cv_auroc(X, y, groups, layer, C, n_splits=5, seed=0):
    """StratifiedGroupKFold CV: standardize on train fold, fit LR, eval on val fold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs, auprcs = [], []
    for tr_idx, va_idx in sgkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr_idx])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr_idx]), y[tr_idx])
        s = clf.decision_function(sc.transform(X[va_idx]))
        aurocs.append(roc_auc_score(y[va_idx], s))
        # AUPRC (average precision)
        from sklearn.metrics import average_precision_score
        auprcs.append(average_precision_score(y[va_idx], s))
    return np.mean(aurocs), np.mean(auprcs)


# ---- M_ref_rep CV: select layer & C on train only ----
cv_rows = []
best = None  # (auroc, auprc, layer, C)
for layer in range(1, N_LAYERS + 1):
    X = Xr_tr[:, layer - 1, :]
    for C in CS:
        a, p = group_cv_auroc(X, y_train, tr_gids, layer, C)
        cv_rows.append([layer, C, a, p])
        if best is None or a > best[0] or (a == best[0] and p > best[1]):
            best = (a, p, layer, C)
    print(f"  layer {layer}: best C mean AUROC={max(r[2] for r in cv_rows if r[0]==layer):.4f}")

with open(D2 / "train_cv_by_layer.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "C", "mean_auroc", "mean_auprc"])
    w.writerows(cv_rows)

best_auroc, best_auprc, best_layer, best_C = best
print(f"SELECTED: layer={best_layer}, C={best_C}, CV AUROC={best_auroc:.4f}, AUPRC={best_auprc:.4f}")
json.dump({"selected_layer": best_layer, "selected_C": best_C,
           "cv_mean_auroc": best_auroc, "cv_mean_auprc": best_auprc},
          open(D2 / "scripts" / "_selected_lr.json", "w"), indent=2)

# ---- frozen probe on dev ----
sc_tr = StandardScaler().fit(Xr_tr[:, best_layer - 1, :])
clf = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced")
clf.fit(sc_tr.transform(Xr_tr[:, best_layer - 1, :]), y_train)
s_dev = clf.decision_function(sc_tr.transform(Xr_dev[:, best_layer - 1, :]))
auroc_dev = roc_auc_score(y_dev, s_dev)
from sklearn.metrics import average_precision_score
auprc_dev = average_precision_score(y_dev, s_dev)
pred_dev = (s_dev > 0).astype(int)
bal_acc = balanced_accuracy_score(y_dev, pred_dev)
print(f"DEV frozen M_ref_rep: AUROC={auroc_dev:.4f} AUPRC={auprc_dev:.4f} bal_acc={bal_acc:.4f}")

# bootstrap CI on dev (group-level, 1000)
n_boot = 1000
boot_auroc, boot_auprc, boot_bal = [], [], []
n_dev = len(y_dev)
for b in range(n_boot):
    idx = RNG.integers(0, n_dev, n_dev)
    if len(np.unique(y_dev[idx])) < 2:
        boot_auroc.append(np.nan); boot_auprc.append(np.nan); boot_bal.append(np.nan)
        continue
    boot_auroc.append(roc_auc_score(y_dev[idx], s_dev[idx]))
    boot_auprc.append(average_precision_score(y_dev[idx], s_dev[idx]))
    boot_bal.append(balanced_accuracy_score(y_dev[idx], (s_dev[idx] > 0).astype(int)))
boot_auroc = np.array(boot_auroc); boot_auprc = np.array(boot_auprc)
ci_auroc = np.nanpercentile(boot_auroc, [2.5, 97.5])
ci_auprc = np.nanpercentile(boot_auprc, [2.5, 97.5])
print(f"  CI AUROC [{ci_auroc[0]:.4f},{ci_auroc[1]:.4f}] AUPRC [{ci_auprc[0]:.4f},{ci_auprc[1]:.4f}]")

# ---- B_surface ----
tok = AutoTokenizer.from_pretrained(MODEL)


def surface_feats(ss_rows):
    rows = []
    for r in ss_rows:
        q = r["question"]; ref = r["reference"]; cand = r["candidate"]
        r_o = ref
        # reference in the SS cell is r_s; candidate is c_s. But features need r_o and r_s separately.
        rows.append(r)
    # We need r_o and r_s for each group, but rows have per-cell data.
    return None


# Surface features need r_o/r_s per group; reconstruct from group's OO and SS rows.
def build_surface(ss_rows, all_rows):
    feats = []
    gids = []
    ss_map = {r["source_group_id"]: r for r in ss_rows if r["cell"] == "SS"}
    oo_map = {r["source_group_id"]: r for r in all_rows if r["cell"] == "OO"}
    for gid in sorted(ss_map):
        ss_r = ss_map[gid]
        oo_r = oo_map.get(gid)
        r_o = oo_r["reference"] if oo_r else None
        r_s = ss_r["reference"]
        q = ss_r["question"]
        if r_o is None:
            continue
        q_tok = len(tok.encode(q))
        ro_tok = len(tok.encode(r_o))
        rs_tok = len(tok.encode(r_s))
        abs_diff = abs(ro_tok - rs_tok)
        q_char = len(q)
        rs_char = len(r_s)
        rs_words = len(r_s.split())
        has_hyphen = 1 if "-" in r_s else 0
        multiword = 1 if rs_words > 1 else 0
        feats.append([q_tok, ro_tok, rs_tok, abs_diff, q_char, rs_char, rs_words, has_hyphen, multiword])
        gids.append(gid)
    return np.array(feats, dtype=float), gids


Xsur_tr, g_tr = build_surface(train_ss, train_rows)
Xsur_dev, g_de = build_surface(dev_ss, dev_rows)
# align with y_train/y_dev order (sorted gids); y arrays follow ss rows order in file.
dev_ss_by_g = {r["source_group_id"]: r for r in dev_ss}
train_ss_by_g = {r["source_group_id"]: r for r in train_ss}
y_tr_al = np.array([1 if train_ss_by_g[g]["predicted_label"] == "B" else 0 for g in g_tr])
y_de_al = np.array([1 if dev_ss_by_g[g]["predicted_label"] == "B" else 0 for g in g_de])

# CV choose C
best_s = None
for C in CS:
    a, p = group_cv_auroc(Xsur_tr, y_tr_al, np.array(g_tr), 0, C)
    if best_s is None or a > best_s[0]:
        best_s = (a, p, C)
_, _, Cs = best_s
sc_s = StandardScaler().fit(Xsur_tr)
clf_s = LogisticRegression(C=Cs, max_iter=2000, class_weight="balanced")
clf_s.fit(sc_s.transform(Xsur_tr), y_tr_al)
s_sur = clf_s.decision_function(sc_s.transform(Xsur_dev))
auroc_s = roc_auc_score(y_de_al, s_sur)
auprc_s = average_precision_score(y_de_al, s_sur)
print(f"B_surface dev: AUROC={auroc_s:.4f} AUPRC={auprc_s:.4f} (C={Cs})")

json.dump({"feature_names": ["question_token_len", "r_o_token_len", "r_s_token_len",
                              "abs_token_len_diff_ro_rs", "question_char_len", "r_s_char_len",
                              "r_s_word_count", "r_s_has_hyphen", "r_s_is_multiword"],
           "selected_C": Cs}, open(D2 / "surface_baseline_spec.json", "w"), indent=2)

# ---- B_decision: -d_raw ----
ss_dev_d = np.array([dev_ss_by_g[g]["d_raw"] for g in g_de])
ss_tr_d = np.array([train_ss_by_g[g]["d_raw"] for g in g_tr])
auroc_bd = roc_auc_score(y_de_al, -ss_dev_d)
auprc_bd = average_precision_score(y_de_al, -ss_dev_d)
print(f"B_decision dev: AUROC={auroc_bd:.4f} AUPRC={auprc_bd:.4f}")

# delta AUPRC M_ref_rep - B_surface bootstrap (group-level paired on dev)
common_g = set(g_de) & set(de_gids)  # gids from surface alignment vs hidden alignment
# Actually hidden X dev gids are de_gids order; surface g_de order sorted. Align both on common gids sorted.
common = sorted(set(de_gids) & set(g_de))
i_hid = [list(de_gids).index(g) for g in common]
i_sur = [list(g_de).index(g) for g in common]
y_c = np.array([1 if dev_ss_by_g[g]["predicted_label"] == "B" else 0 for g in common])
s_h_c = s_dev[i_hid]
s_s_c = s_sur[i_sur]
d_auprc_boot = []
for b in range(n_boot):
    idx = RNG.integers(0, len(common), len(common))
    if len(np.unique(y_c[idx])) < 2:
        d_auprc_boot.append(np.nan); continue
    a1 = average_precision_score(y_c[idx], s_h_c[idx])
    a2 = average_precision_score(y_c[idx], s_s_c[idx])
    d_auprc_boot.append(a1 - a2)
d_auprc_boot = np.array(d_auprc_boot)
ci_d = np.nanpercentile(d_auprc_boot, [2.5, 97.5])
print(f"ΔAUPRC(M-B_surface) dev CI [{ci_d[0]:.4f},{ci_d[1]:.4f}]")

# ---- permutation null (200) ----
perm_aurocs = []
base_idx = np.arange(len(y_train))
for p_i in range(200):
    yp = y_train[RNG.permutation(len(y_train))]
    sc_p = StandardScaler().fit(Xr_tr[:, best_layer - 1, :])
    clf_p = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced")
    clf_p.fit(sc_p.transform(Xr_tr[:, best_layer - 1, :]), yp)
    s_p = clf_p.decision_function(sc_p.transform(Xr_dev[:, best_layer - 1, :]))
    perm_aurocs.append(roc_auc_score(y_dev, s_p))
perm_aurocs = np.array(perm_aurocs)
p975 = np.percentile(perm_aurocs, 97.5)
print(f"permutation null 97.5% = {p975:.4f}, real dev AUROC={auroc_dev:.4f}, sig={auroc_dev > p975}")

# ---- four-cell contrasts ----
cells = ["OO", "OS", "SO", "SS"]
contrast_rows = []
for split, hid, ss_rows, gids_l, yl in (("train", train_hid, train_ss, tr_gids, y_train),
                                        ("dev", dev_hid, dev_ss, de_gids, y_dev)):
    # compute Δ for each group at each position & layer
    for gi, g in enumerate(gids_l):
        h = hid[g]
        for pos in ("r", "c", "d"):
            hSS = h["SS"][pos].astype(np.float32)
            hOS = h["OS"][pos].astype(np.float32)
            hSO = h["SO"][pos].astype(np.float32)
            hOO = h["OO"][pos].astype(np.float32)
            d_ref = hSS - hOS
            d_cand = hSS - hSO
            d_int = hSS - hSO - hOS + hOO
            for li in range(N_LAYERS):
                contrast_rows.append([split, g, pos, li + 1, yl[gi],
                                      np.linalg.norm(d_ref[li]), np.linalg.norm(d_cand[li]),
                                      np.linalg.norm(d_int[li])])

with open(D2 / "four_cell_representation_contrasts.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "group_id", "position", "layer", "y_SS_error",
                "L2_dref", "L2_dcand", "L2_dint"])
    w.writerows(contrast_rows)
print("wrote four_cell_representation_contrasts.csv")

# ---- decision_score_diagnostic.csv ----
with open(D2 / "decision_score_diagnostic.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value", "note"])
    w.writerow(["AUROC", auroc_bd, "uses final judge decision (-d_raw); NOT a fair comparison to M_ref_rep"])
    w.writerow(["AUPRC", auprc_bd, "upper-bound diagnostic only"])

# ---- metrics_primary_dev.csv ----
with open(D2 / "metrics_primary_dev.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value", "ci_lower_95", "ci_upper_95"])
    w.writerow(["M_ref_rep_AUROC", auroc_dev, ci_auroc[0], ci_auroc[1]])
    w.writerow(["M_ref_rep_AUPRC", auprc_dev, ci_auprc[0], ci_auprc[1]])
    w.writerow(["M_ref_rep_balanced_acc", bal_acc, "", ""])
    w.writerow(["B_surface_AUROC", auroc_s, "", ""])
    w.writerow(["B_surface_AUPRC", auprc_s, "", ""])
    w.writerow(["Delta_AUPRC_Mref_minus_Bsurface", auprc_dev - auprc_s, ci_d[0], ci_d[1]])
    w.writerow(["selected_layer", best_layer, "", ""])
    w.writerow(["selected_C", best_C, "", ""])

# ---- permutation_null_audit.csv ----
with open(D2 / "permutation_null_audit.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_perm", "real_dev_auroc", "null_p97_5", "significant"])
    w.writerow([200, auroc_dev, p975, auroc_dev > p975])

print("=== KEY METRICS ===")
print(json.dumps({"M_ref_rep_dev_AUROC": auroc_dev, "M_ref_rep_dev_AUPRC": auprc_dev,
                  "CI_AUROC": ci_auroc.tolist(), "CI_AUPRC": ci_auprc.tolist(),
                  "B_surface_dev_AUROC": auroc_s, "B_surface_dev_AUPRC": auprc_s,
                  "Delta_AUPRC_CI": ci_d.tolist(), "B_decision_AUROC": auroc_bd,
                  "B_decision_AUPRC": auprc_bd, "perm_p97_5": p975}, indent=2))
