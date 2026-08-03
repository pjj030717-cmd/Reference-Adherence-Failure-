#!/usr/bin/env python3
"""E01-D2: D1-R template transfer diagnostic (T1/T2) on dev SS inputs.

1. Re-score dev SS inputs under T1/T2; verify aggregate against D1-R:
   T1 SS accuracy = 0.169, T2 SS accuracy = 0.113
2. Verify R_end hidden states identical across T0/T1/T2 (prefix_causality_audit).
3. Frozen M_ref_rep risk scores on T1/T2 SS labels: report AUROC/AUPRC.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL, ACCEPT, REJECT = CONST["system"], CONST["user_template"], CONST["accept"], CONST["reject"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]
T1 = "For this question, the answer is <answer>."
T2 = "The response is <answer>."
SEL = json.loads((D2 / "scripts" / "_selected_lr.json").read_text(encoding="utf-8"))
L = SEL["selected_layer"]
C = SEL["selected_C"]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()

dev_ss = [r for r in json.loads((D2 / "scripts" / "_dev_rows.json").read_text(encoding="utf-8"))
          if r["cell"] == "SS"]
print("dev SS:", len(dev_ss))

# load T0 R_end for comparison
dev_gids = [r["source_group_id"] for r in dev_ss]
t0_r = np.stack([np.load(D2 / "hidden_states" / f"dev_{g}.npz")["SS_h_r"] for g in dev_gids]).astype(np.float32)  # (n,28,d)


def score_template(template):
    acc = []
    r_end_cols = []
    for r in dev_ss:
        q, ref, cand = r["question"], r["reference"], r["candidate"].replace(
            "The answer is ", "__PLACEHOLDER__")
        cand = template.replace("<answer>", ref)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(question=q, reference=ref, candidate=cand)},
        ]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        # locate reference end
        ref_marker = "Reference Answer: "
        i0 = rendered.find(ref_marker)
        ref_start = i0 + len(ref_marker)
        ref_end = ref_start + len(ref)
        r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
        prompt_ids = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            out = model(prompt_ids, output_hidden_states=True)
            logits = out.logits
            hs = out.hidden_states
        pos = prompt_ids.shape[1] - 1
        l_A = logits[0, pos, ACCEPT_ID].item()
        l_B = logits[0, pos, REJECT_ID].item()
        d_raw = l_A - l_B
        pred = "A" if d_raw > 0 else ("B" if d_raw < 0 else "TIE")
        acc.append(pred == "A")
        r_end_cols.append(np.stack([hs[l][0, r_tok, :].cpu().float().numpy() for l in range(1, 29)]))

    agg = np.mean(acc)
    return agg, np.stack(r_end_cols).astype(np.float32), acc


results = {}
for name, t in (("T1", T1), ("T2", T2)):
    agg, r_end, acc = score_template(t)
    # prefix audit: R_end must equal T0 R_end
    max_diff = np.abs(r_end - t0_r).max()
    mean_diff = np.abs(r_end - t0_r).mean()
    print(f"{name}: SS acc={agg:.4f} (D1-R: 0.169/0.113), R_end max diff vs T0={max_diff:.6f} mean={mean_diff:.6f}")
    results[name] = {"ss_accuracy": agg, "r_end_max_diff": float(max_diff), "r_end_mean_diff": float(mean_diff)}

# D1-R aggregates to match
d1r_exp = {"T1": 0.169, "T2": 0.113}
audit_ok = True
for name in ("T1", "T2"):
    if abs(results[name]["ss_accuracy"] - d1r_exp[name]) > 0.01:
        print("D1-R aggregate not reproduced for", name)
        audit_ok = False
    if results[name]["r_end_max_diff"] > 1e-4:
        print(f"R_end differs for {name}: max diff {results[name]['r_end_max_diff']:.4f}")
        audit_ok = False
print("prefix_causality_audit:", "PASSED" if audit_ok else "INVALID")

# frozen probe on T1/T2 labels
tr_rows = json.loads((D2 / "scripts" / "_train_rows.json").read_text(encoding="utf-8"))
tr_ss = [r for r in tr_rows if r["cell"] == "SS"]
tr_gids = [r["source_group_id"] for r in tr_ss]
Xr_tr = np.stack([np.load(D2 / "hidden_states" / f"train_{g}.npz")["SS_h_r"] for g in tr_gids]).astype(np.float32)
y_tr = np.array([1 if r["predicted_label"] == "B" else 0 for r in tr_ss])

sc = StandardScaler().fit(Xr_tr[:, L - 1, :])
clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
clf.fit(sc.transform(Xr_tr[:, L - 1, :]), y_tr)

# for T1/T2 we need their own labels (pred B = error). We already computed acc but not per-group labels.
# Re-score labels are deterministic from d_raw; recompute quickly using saved agg? We need y per group.
# Re-run scoring to capture per-group predictions.
def labels_template(template):
    labs = []
    for r in dev_ss:
        q, ref = r["question"], r["reference"]
        cand = template.replace("<answer>", ref)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(question=q, reference=ref, candidate=cand)},
        ]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        prompt_ids = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            logits = model(prompt_ids).logits
        pos = prompt_ids.shape[1] - 1
        d_raw = logits[0, pos, ACCEPT_ID].item() - logits[0, pos, REJECT_ID].item()
        labs.append(1 if d_raw < 0 else 0)
    return np.array(labs)

# R_end risk scores (from T0 R_end == frozen probe input; same as dev T0)
s_r = clf.decision_function(sc.transform(t0_r[:, L - 1, :]))

rows = []
for name, t in (("T1", T1), ("T2", T2)):
    y_t = labels_template(t)
    a = roc_auc_score(y_t, s_r)
    p = average_precision_score(y_t, s_r)
    rows.append([name, results[name]["ss_accuracy"], len(y_t), int(y_t.sum()),
                 a, p, results[name]["r_end_max_diff"]])
    print(f"{name}: y=1 n={y_t.sum()} AUROC={a:.4f} AUPRC={p:.4f}")

with open(D2 / "metrics_template_transfer_dev.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["template", "ss_accuracy", "n_ss", "n_error", "AUROC_frozen_Rend", "AUPRC_frozen_Rend", "R_end_max_diff_vs_T0"])
    w.writerows(rows)
print("wrote metrics_template_transfer_dev.csv")
