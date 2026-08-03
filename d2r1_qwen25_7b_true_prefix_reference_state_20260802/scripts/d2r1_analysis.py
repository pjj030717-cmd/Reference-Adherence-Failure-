#!/usr/bin/env python3
"""D2-R1: main analysis.
- ss_label_capacity_audit.csv
- train_cv_by_layer.csv (M_true_prefix_rep layer/C selection on train only)
- metrics_true_prefix_dev.csv (frozen probe on dev + bootstrap CI)
- surface_baseline_metrics.csv (B_surface)
- metrics_template_transfer_dev.csv (T1/T2 frozen transfer)
- permutation_null_audit.csv (200 perms)
- d2_invalid_result_boundary_note.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"
D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
CS = [0.0001, 0.001, 0.01, 0.1, 1.0]
RNG = np.random.default_rng(20260802)
N_LAYERS = 28

dev_ss = json.loads((D2R1 / "scripts" / "_ss_dev_scores.json").read_text(encoding="utf-8"))
train_ss = json.loads((D2R1 / "scripts" / "_ss_train_scores.json").read_text(encoding="utf-8"))


def ss_y(rows):
    return np.array([1 if r["predicted_label"] == "B" else 0 for r in rows])


y_dev = ss_y(dev_ss)
y_train = ss_y(train_ss)
print("train SS n:", len(y_train), "y=1:", y_train.sum(), "y=0:", (1 - y_train).sum())
print("dev   SS n:", len(y_dev), "y=1:", y_dev.sum(), "y=0:", (1 - y_dev).sum())

# ---- capacity audit ----
with open(D2R1 / "ss_label_capacity_audit.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "n_ss", "n_error(y=1)", "n_correct(y=0)", "gate"])
    w.writerow(["train", len(y_train), int(y_train.sum()), int((1 - y_train).sum()),
                "PASS" if y_train.sum() >= 100 and (1 - y_train).sum() >= 100 else "FAIL"])
    w.writerow(["dev", len(y_dev), int(y_dev.sum()), int((1 - y_dev).sum()),
                "PASS" if y_dev.sum() >= 30 and (1 - y_dev).sum() >= 30 else "FAIL"])
if not (y_train.sum() >= 100 and (1 - y_train).sum() >= 100 and y_dev.sum() >= 30 and (1 - y_dev).sum() >= 30):
    print("STOP: ss_outcome_capacity_insufficient")
    (D2R1 / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": "ss_outcome_capacity_insufficient", "d2_hidden_arrays_reused": False,
         "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
         "probe_trained": True, "activation_intervention_run": False,
         "prompt_baselines_run": False, "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)
print("capacity gate PASS")


def load_X(rows):
    return np.stack([np.load(D2R1 / "prefix_hidden_states" /
                            f"{'dev' if rows[0]['source_group_id'] in {r['source_group_id'] for r in dev_ss} else 'train'}_{r['source_group_id']}.npz")["h_prefix"]
                     for r in rows]).astype(np.float32)


Xr_tr = load_X(train_ss)  # (587, 28, 3584)
Xr_dev = load_X(dev_ss)   # (195, 28, 3584)
tr_gids = np.array([r["source_group_id"] for r in train_ss])
de_gids = np.array([r["source_group_id"] for r in dev_ss])


def group_cv(X, y, groups, C, n_splits=5):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=20260802)
    aurocs, auprcs = [], []
    for tr_idx, va_idx in sgkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr_idx])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr_idx]), y[tr_idx])
        s = clf.decision_function(sc.transform(X[va_idx]))
        aurocs.append(roc_auc_score(y[va_idx], s))
        auprcs.append(average_precision_score(y[va_idx], s))
    return np.mean(aurocs), np.mean(auprcs)


# ---- M_true_prefix_rep: select layer then C, train only ----
cv_rows = []
best = None
for layer in range(1, N_LAYERS + 1):
    X = Xr_tr[:, layer - 1, :]
    for C in CS:
        a, p = group_cv(X, y_train, tr_gids, C)
        cv_rows.append([layer, C, a, p])
        if best is None or a > best[0] or (a == best[0] and p > best[1]):
            best = (a, p, layer, C)
    print(f"  layer {layer}: best mean AUROC={max(r[2] for r in cv_rows if r[0]==layer):.4f}")

with open(D2R1 / "train_cv_by_layer.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "C", "mean_auroc", "mean_auprc"])
    w.writerows(cv_rows)

best_auroc, best_auprc, best_layer, best_C = best
print(f"SELECTED: layer={best_layer}, C={best_C}, CV AUROC={best_auroc:.4f}, AUPRC={best_auprc:.4f}")
json.dump({"selected_layer": best_layer, "selected_C": best_C,
           "cv_mean_auroc": best_auroc, "cv_mean_auprc": best_auprc},
          open(D2R1 / "scripts" / "_selected_lr.json", "w"), indent=2)

# ---- frozen probe on dev ----
sc_tr = StandardScaler().fit(Xr_tr[:, best_layer - 1, :])
clf = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced")
clf.fit(sc_tr.transform(Xr_tr[:, best_layer - 1, :]), y_train)
s_dev = clf.decision_function(sc_tr.transform(Xr_dev[:, best_layer - 1, :]))
auroc_dev = roc_auc_score(y_dev, s_dev)
auprc_dev = average_precision_score(y_dev, s_dev)
bal_acc = balanced_accuracy_score(y_dev, (s_dev > 0).astype(int))
print(f"DEV frozen M_true_prefix_rep: AUROC={auroc_dev:.4f} AUPRC={auprc_dev:.4f} bal_acc={bal_acc:.4f}")

n_boot = 1000
n_dev = len(y_dev)
boot = np.array([[roc_auc_score(y_dev[idx], s_dev[idx])
                  if len(np.unique(y_dev[idx])) >= 2 else np.nan,
                  average_precision_score(y_dev[idx], s_dev[idx])
                  if len(np.unique(y_dev[idx])) >= 2 else np.nan]
                 for idx in (RNG.integers(0, n_dev, n_dev) for _ in range(n_boot))])
ci_auroc = np.nanpercentile(boot[:, 0], [2.5, 97.5])
ci_auprc = np.nanpercentile(boot[:, 1], [2.5, 97.5])
print(f"  CI AUROC [{ci_auroc[0]:.4f},{ci_auroc[1]:.4f}] AUPRC [{ci_auprc[0]:.4f},{ci_auprc[1]:.4f}]")

# ---- B_surface ----
tok = AutoTokenizer.from_pretrained(MODEL)


def build_surface(rows):
    feats, gids = [], []
    for r in rows:
        q = r["question"]
        r_s = r["reference"]
        # r_o from D2 row (OO reference) -- fetch from D2 dev/train rows
        pool = json.loads((D2 / "scripts" / "_dev_rows.json").read_text(encoding="utf-8")) \
            if r["source_group_id"] in {x["source_group_id"] for x in dev_ss} else \
            json.loads((D2 / "scripts" / "_train_rows.json").read_text(encoding="utf-8"))
        oo = next((x for x in pool if x["source_group_id"] == r["source_group_id"] and x["cell"] == "OO"), None)
        r_o = oo["reference"] if oo else None
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
        gids.append(r["source_group_id"])
    return np.array(feats, dtype=float), gids


Xsur_tr, g_tr = build_surface(train_ss)
Xsur_dev, g_de = build_surface(dev_ss)
y_tr_al = np.array([1 if next(r for r in train_ss if r["source_group_id"] == g)["predicted_label"] == "B" else 0
                    for g in g_tr])
y_de_al = np.array([1 if next(r for r in dev_ss if r["source_group_id"] == g)["predicted_label"] == "B" else 0
                    for g in g_de])

best_s = None
for C in CS:
    a, p = group_cv(Xsur_tr, y_tr_al, np.array(g_tr), C)
    if best_s is None or a > best_s[0]:
        best_s = (a, p, C)
Cs = best_s[2]
sc_s = StandardScaler().fit(Xsur_tr)
clf_s = LogisticRegression(C=Cs, max_iter=2000, class_weight="balanced")
clf_s.fit(sc_s.transform(Xsur_tr), y_tr_al)
s_sur = clf_s.decision_function(sc_s.transform(Xsur_dev))
auroc_s = roc_auc_score(y_de_al, s_sur)
auprc_s = average_precision_score(y_de_al, s_sur)
print(f"B_surface dev: AUROC={auroc_s:.4f} AUPRC={auprc_s:.4f} (C={Cs})")

with open(D2R1 / "surface_baseline_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value", "note"])
    w.writerow(["B_surface_AUROC", auroc_s, "dev frozen logistic (train-CV-selected C)"])
    w.writerow(["B_surface_AUPRC", auprc_s, "dev"])
    w.writerow(["selected_C", Cs, "train 5-fold group CV"])
    w.writerow(["feature_count", 9, "pre-registered model-free surface features"])

# ---- ΔAUPRC(M - B_surface) bootstrap (common groups) ----
common = sorted(set(de_gids) & set(g_de))
i_h = [list(de_gids).index(g) for g in common]
i_s = [list(g_de).index(g) for g in common]
y_c = y_dev[i_h]
s_h = s_dev[i_h]
s_b = s_sur[i_s]
d_auprc = np.array([
    average_precision_score(y_c[idx], s_h[idx]) - average_precision_score(y_c[idx], s_b[idx])
    if len(np.unique(y_c[idx])) >= 2 else np.nan
    for idx in (RNG.integers(0, len(common), len(common)) for _ in range(n_boot))])
ci_d = np.nanpercentile(d_auprc, [2.5, 97.5])
print(f"ΔAUPRC(M-B_surface) CI [{ci_d[0]:.4f},{ci_d[1]:.4f}]")

# ---- permutation null (200) ----
perm = []
for p_i in range(200):
    yp = y_train[RNG.permutation(len(y_train))]
    sc_p = StandardScaler().fit(Xr_tr[:, best_layer - 1, :])
    clf_p = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced")
    clf_p.fit(sc_p.transform(Xr_tr[:, best_layer - 1, :]), yp)
    s_p = clf_p.decision_function(sc_p.transform(Xr_dev[:, best_layer - 1, :]))
    perm.append(roc_auc_score(y_dev, s_p))
perm = np.array(perm)
p975 = np.percentile(perm, 97.5)
print(f"permutation null 97.5%={p975:.4f}, real dev AUROC={auroc_dev:.4f}, sig={auroc_dev > p975}")

with open(D2R1 / "permutation_null_audit.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_perm", "real_dev_auroc", "null_p97_5", "significant"])
    w.writerow([200, auroc_dev, p975, auroc_dev > p975])

# ---- metrics_true_prefix_dev.csv ----
with open(D2R1 / "metrics_true_prefix_dev.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value", "ci_lower_95", "ci_upper_95"])
    w.writerow(["M_true_prefix_rep_AUROC", auroc_dev, ci_auroc[0], ci_auroc[1]])
    w.writerow(["M_true_prefix_rep_AUPRC", auprc_dev, ci_auprc[0], ci_auprc[1]])
    w.writerow(["M_true_prefix_rep_balanced_acc", bal_acc, "", ""])
    w.writerow(["B_surface_AUROC", auroc_s, "", ""])
    w.writerow(["B_surface_AUPRC", auprc_s, "", ""])
    w.writerow(["Delta_AUPRC_M_minus_B", auprc_dev - auprc_s, ci_d[0], ci_d[1]])
    w.writerow(["selected_layer", best_layer, "", ""])
    w.writerow(["selected_C", best_C, "", ""])

# ---- T1/T2 frozen transfer ----
import torch
from transformers import AutoModelForCausalLM
T1 = "For this question, the answer is <answer>."
T2 = "The response is <answer>."
CONST1 = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM1, USER_TMPL1 = CONST1["system"], CONST1["user_template"]
ACCEPT_ID, REJECT_ID = CONST1["accept_id"], CONST1["reject_id"]

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()


def score_ss(template):
    labels = []
    for r in dev_ss:
        q, ref = r["question"], r["reference"]
        cand = template.replace("<answer>", ref)
        messages = [{"role": "system", "content": SYSTEM1},
                    {"role": "user", "content": USER_TMPL1.format(question=q, reference=ref, candidate=cand)}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        pids = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            logits = model(pids).logits
        pos = pids.shape[1] - 1
        d_raw = logits[0, pos, ACCEPT_ID].item() - logits[0, pos, REJECT_ID].item()
        labels.append(1 if d_raw < 0 else 0)
    return np.array(labels)


d1r_exp = {"T1": 0.169, "T2": 0.113}
t_rows = []
for name, t in (("T1", T1), ("T2", T2)):
    y_t = score_ss(t)
    acc = float(1 - y_t.mean())  # error rate -> accuracy = P(predict A)
    a = roc_auc_score(y_t, s_dev)
    p = average_precision_score(y_t, s_dev)
    t_rows.append([name, acc, len(y_t), int(y_t.sum()), a, p,
                   "OK" if abs(acc - d1r_exp[name]) <= 0.01 else "MISMATCH"])
    print(f"{name}: SS acc={acc:.4f} (D1-R {d1r_exp[name]}) n_err={y_t.sum()} AUROC={a:.4f} AUPRC={p:.4f}")

with open(D2R1 / "metrics_template_transfer_dev.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["template", "ss_accuracy", "n_ss", "n_error", "AUROC_frozen", "AUPRC_frozen", "d1r_match"])
    w.writerows(t_rows)

# ---- d2_invalid_result_boundary_note.md ----
(D2R1 / "d2_invalid_result_boundary_note.md").write_text(
    """# d2_invalid_result_boundary_note.md

## D2 无效点与本轮修正的边界说明

D2（`prefix_causality_audit_invalid`）的无效点：

- D2 在完整 `Question + Reference + Candidate + Answer:` 序列上做前向，事后读取 Reference 末尾 token 的 hidden state。
- 该状态随后续 Candidate 文本的**序列总长度**产生数值变化（eager 与 sdpa 均复现；截断到前缀后与 T0 一致）。
- 因此 D2 的 R_end 不是严格"纯前缀"状态，D2 的结论被预注册协议判为无效。

D2-R1 的修正：

- 输入在 Reference Answer 最后一个 token 处**真正停止**：`prefix_input_ids = full_input_ids[:R_end+1]`。
- prefix 之后不存在 Candidate Answer、`Answer:`、generation prompt、padding 或 suffix token。
- `h_prefix[layer] = hidden_states[layer][0, prefix_len-1, :]` 只来自这个截断前向。

边界（不得越界解释）：

- 本轮的"合同"是：**相同截断输入重复运行的确定性**，以及 **T0/T1/T2 前缀 token ids 完全相同**。
- 不再要求完整长序列中早期 token 的 hidden state 与截断前向一致（这正是 D2 失效的机制）。
- D2 的正式标签 `prefix_causality_audit_invalid` 原样保留，不得覆写。
- D2 的完整序列 R_end 结果仅作"已作废的诊断"在报告中引用，不进入任何比较表、不合并指标、不作为正式证据。
""",
    encoding="utf-8")

print("=== KEY METRICS ===")
print(json.dumps({"M_true_prefix_dev_AUROC": auroc_dev, "M_true_prefix_dev_AUPRC": auprc_dev,
                  "CI_AUROC": ci_auroc.tolist(), "CI_AUPRC": ci_auprc.tolist(),
                  "B_surface_dev_AUROC": auroc_s, "B_surface_dev_AUPRC": auprc_s,
                  "Delta_AUPRC_CI": ci_d.tolist(), "perm_p97_5": p975}, indent=2))
