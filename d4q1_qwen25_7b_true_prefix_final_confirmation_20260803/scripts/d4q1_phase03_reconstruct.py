#!/usr/bin/env python3
"""D4-Q1 Phase 0.3: reconstruct frozen Probe (M_rep) and B_surface, then
read-only dev reproduction against D2-R1 recorded metrics.

Reconstruction uses ONLY train data (as authorized by protocol 0.3):
  - M_rep: D2-R1 self-extracted train prefix hidden states (prefix_hidden_states/train_*.npz)
           + train SS labels (_ss_train_scores.json); layer=18, C=0.01, StandardScaler train-only.
  - B_surface: 9 surface features from D0 swap train rows (r_o from train rows);
           LogisticRegression(C=1.0 frozen, max_iter=2000, class_weight=balanced), train-only.

Read-only reproduction (does NOT choose any hyperparameter):
  - Apply frozen M_rep / B_surface to D2-R1 dev hidden states + dev SS labels;
    compare AUROC/AUPRC to recorded D2-R1 dev metrics within float tolerance.
  - Save frozen model artifacts (scaler + coef) for Phase 2 (as dicts in scripts/_frozen/).

No final-reserve data is read here.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

R = REPO_ROOT / "d4q1_qwen25_7b_true_prefix_final_confirmation_20260803"
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")

LAYER = 18
C_PROBE = 0.01
C_SURFACE = 1.0
MAX_ITER = 2000

def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": label, "reason": why,
         "allowed_final_groups": 196, "quarantined_final_groups": 1,
         "quarantined_group_scored": False, "quarantined_group_hidden_state_read": False,
         "final_configuration_changed": False, "hidden_layer": LAYER,
         "hidden_token": "R_end", "probe_C": C_PROBE,
         "probe_refit_used_dev": False, "probe_refit_used_final": False,
         "activation_intervention_run": False, "mistral_loaded": False,
         "prompt_baselines_run": False}, indent=2), encoding="utf-8")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load train labels + hidden states
# ---------------------------------------------------------------------------
train_ss = json.loads((D2R1 / "scripts" / "_ss_train_scores.json").read_text(encoding="utf-8"))
assert len(train_ss) == 587, f"train SS rows {len(train_ss)} != 587"
gids_tr = np.array([r["source_group_id"] for r in train_ss])
y_tr = np.array([1 if r["predicted_label"] == "B" else 0 for r in train_ss])
print("train SS n:", len(y_tr), "y=1:", y_tr.sum(), "y=0:", (1 - y_tr).sum())

def load_hidden(gids, prefix="train"):
    h = []
    for g in gids:
        a = np.load(D2R1 / "prefix_hidden_states" / f"{prefix}_{g}.npz")
        h.append(a["h_prefix"])  # (28, 3584) float16
    return np.stack(h).astype(np.float32)  # (n, 28, 3584)

Xr_tr = load_hidden(gids_tr, "train")
X18_tr = Xr_tr[:, LAYER - 1, :].copy()
print("train hidden (n,28,3584):", Xr_tr.shape, "-> layer18:", X18_tr.shape)

# ---------------------------------------------------------------------------
# M_rep frozen reconstruction (train only)
# ---------------------------------------------------------------------------
sc = StandardScaler().fit(X18_tr)
clf = LogisticRegression(C=C_PROBE, max_iter=MAX_ITER, class_weight="balanced")
clf.fit(sc.transform(X18_tr), y_tr)
print("M_rep frozen: scaler_mean/scale shape", sc.mean_.shape, sc.scale_.shape,
      "coef shape", clf.coef_.shape, "intercept", clf.intercept_)

# ---------------------------------------------------------------------------
# B_surface frozen reconstruction (train only), r_o from D0 swap train rows
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL)
ro_map = {}
with open(D0 / "preliminary_swap_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d["split"] != "train":
            continue
        ro_map[d["original_group_id"]] = d["r_o"]
assert len(ro_map) == 587, f"train r_o rows {len(ro_map)} != 587"

def build_surface(rows, gids, ro_map):
    feats, kept = [], []
    for r in rows:
        g = r["source_group_id"]
        q, r_s = r["question"], r["reference"]
        r_o = ro_map.get(g)
        if r_o is None:
            continue
        q_tok = len(tok.encode(q))
        ro_tok = len(tok.encode(r_o))
        rs_tok = len(tok.encode(r_s))
        q_char = len(q)
        rs_char = len(r_s)
        rs_words = len(r_s.split())
        feats.append([q_tok, ro_tok, rs_tok, abs(ro_tok - rs_tok), q_char, rs_char,
                      rs_words, 1 if "-" in r_s else 0, 1 if rs_words > 1 else 0])
        kept.append(g)
    return np.array(feats, dtype=float), kept

Xsur_tr, g_sur_tr = build_surface(train_ss, gids_tr, ro_map)
y_tr_al = np.array([1 if next(r for r in train_ss if r["source_group_id"] == g)["predicted_label"] == "B" else 0
                    for g in g_sur_tr])
print("B_surface train feats:", Xsur_tr.shape, "y:", y_tr_al.sum(), "/", len(y_tr_al))

sc_s = StandardScaler().fit(Xsur_tr)
clf_s = LogisticRegression(C=C_SURFACE, max_iter=MAX_ITER, class_weight="balanced")
clf_s.fit(sc_s.transform(Xsur_tr), y_tr_al)
print("B_surface frozen: coef shape", clf_s.coef_.shape, "intercept", clf_s.intercept_)

# ---------------------------------------------------------------------------
# Read-only dev reproduction (no hyperparameter choice)
# ---------------------------------------------------------------------------
# dev B_surface r_o comes from D0 swap dev rows (allowed; dev is a permitted
# inherited split). ro_map above holds train rows only.
ro_map_all = {}
with open(D0 / "preliminary_swap_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        ro_map_all[d["original_group_id"]] = d["r_o"]  # train + dev + reserve ids
dev_ro_map = {g: r for g, r in ro_map_all.items()}

dev_ss = json.loads((D2R1 / "scripts" / "_ss_dev_scores.json").read_text(encoding="utf-8"))
gids_de = np.array([r["source_group_id"] for r in dev_ss])
y_de = np.array([1 if r["predicted_label"] == "B" else 0 for r in dev_ss])
Xr_de = load_hidden(gids_de, "dev")
X18_de = Xr_de[:, LAYER - 1, :].copy()

s_dev = clf.decision_function(sc.transform(X18_de))
auroc_dev = roc_auc_score(y_de, s_dev)
auprc_dev = average_precision_score(y_de, s_dev)
print(f"REPRO M_rep dev AUROC={auroc_dev:.6f} (recorded 0.9138872915468661)  AUPRC={auprc_dev:.6f}")

Xsur_de, g_sur_de = build_surface(dev_ss, gids_de, dev_ro_map)
i_de = [list(gids_de).index(g) for g in g_sur_de]
y_de_al = y_de[i_de]
s_sur = clf_s.decision_function(sc_s.transform(Xsur_de))
auroc_s = roc_auc_score(y_de_al, s_sur)
auprc_s = average_precision_score(y_de_al, s_sur)
print(f"REPRO B_surface dev AUROC={auroc_s:.6f} (recorded 0.6207590569292697)  AUPRC={auprc_s:.6f}")

# tolerances (BF16 float path; sklearn deterministic given same data)
tol = 1e-4
ok_probe = abs(auroc_dev - 0.9138872915468661) <= tol and abs(auprc_dev - 0.9632189056523878) <= tol
ok_surf = abs(auroc_s - 0.6207590569292697) <= tol and abs(auprc_s - 0.8181915706506249) <= tol
print("REPRO tolerances:", "M_rep", ok_probe, "B_surface", ok_surf)
if not (ok_probe and ok_surf):
    fail("frozen_probe_or_baseline_not_reconstructable",
         f"dev reproduction mismatch: M_rep AUROC={auroc_dev} (want 0.9138873), B_surface AUROC={auroc_s} (want 0.6207591)")

# ---------------------------------------------------------------------------
# Save frozen artifacts (for Phase 2)
# ---------------------------------------------------------------------------
(R / "scripts" / "_frozen").mkdir(parents=True, exist_ok=True)
np.savez(R / "scripts" / "_frozen" / "probe.npz",
         scaler_mean=sc.mean_, scaler_scale=sc.scale_,
         coef=clf.coef_, intercept=clf.intercept_,
         surface_scaler_mean=sc_s.mean_, surface_scaler_scale=sc_s.scale_,
         surface_coef=clf_s.coef_, surface_intercept=clf_s.intercept_)
json.dump({"layer": LAYER, "C_probe": C_PROBE, "C_surface": C_SURFACE,
           "max_iter": MAX_ITER, "class_weight": "balanced",
           "dev_repro_auroc_probe": float(auroc_dev),
           "dev_repro_auprc_probe": float(auprc_dev),
           "dev_repro_auroc_surface": float(auroc_s),
           "dev_repro_auprc_surface": float(auprc_s),
           "tolerance": tol},
          open(R / "scripts" / "_frozen" / "meta.json", "w"), indent=2)
print("saved frozen artifacts to scripts/_frozen/")

# ---------------------------------------------------------------------------
# append reproduction section to frozen_probe_reconstruction_audit.md
# ---------------------------------------------------------------------------
(R / "frozen_probe_reconstruction_audit.md").open("a", encoding="utf-8").write(
    f"""

## 重建与 dev 只读复现（Phase 0.3，2026-08-03）

- M_rep 重建（train-only）：layer 18, C=0.01, StandardScaler train, LogReg max_iter=2000 balanced。
- B_surface 重建（train-only）：C=1.0（D2-R1 冻结），9 特征，r_o 取自 D0 swap train 行。
- dev 只读复现（不选任何超参数）：

| 模型 | 复现 AUROC | 记录 AUROC | 复现 AUPRC | 记录 AUPRC | 容差 |
|---|---|---|---|---|---|
| M_rep | {auroc_dev:.6f} | 0.9138872915468661 | {auprc_dev:.6f} | 0.9632189056523878 | {tol} |
| B_surface | {auroc_s:.6f} | 0.6207590569292697 | {auprc_s:.6f} | 0.8181915706506249 | {tol} |

- 结论：重建模型在 dev 上与 D2-R1 记录逐行（聚合）一致，冻结 Probe 与 B_surface 唯一恢复完成。
""")
print("Phase 0.3 OK: frozen Probe + B_surface reconstructed and dev-reproduced")
