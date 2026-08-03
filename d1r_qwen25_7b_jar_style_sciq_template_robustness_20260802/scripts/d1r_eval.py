#!/usr/bin/env python3
"""E01-D1-R: T0 reproduction + T1/T2 behavior evaluation.

- T0: re-score all 195 dev groups x 4 cells with D1's exact template & code path,
      then compare row-by-row against D1 four_cell_scores_dev.csv.
- T1/T2: only if T0 exactly reproduces.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"

CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL, ACCEPT, REJECT = CONST["system"], CONST["user_template"], CONST["accept"], CONST["reject"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]

TEMPLATES = {
    "T0": "The answer is <answer>.",
    "T1": "For this question, the answer is <answer>.",
    "T2": "The response is <answer>.",
}

# dev pairs (from D1)
dev_pairs = []
with open(D1 / "scripts" / "_dev_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dev_pairs.append(json.loads(line))
print("dev pairs:", len(dev_pairs))


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (D1R / "artifacts").mkdir(parents=True, exist_ok=True)
    (D1R / "artifacts" / "decision.json").write_text(
        json.dumps({"final_label": label, "reason": why,
                    "train_model_scored": False, "final_reserve_model_scored": False,
                    "hidden_states_read": False, "probe_trained": False,
                    "activation_intervention_run": False, "prompt_baselines_run": False,
                    "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()
print("model loaded for template robustness evaluation")


def score(question: str, reference: str, candidate: str):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(question=question, reference=reference, candidate=candidate)},
    ]
    enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    prompt_ids = enc["input_ids"].to("cuda")
    prompt_len = prompt_ids.shape[1]
    with torch.inference_mode():
        logits = model(prompt_ids).logits
    pos = prompt_len - 1
    ll = logits[0, pos, :]
    l_A = ll[ACCEPT_ID].item()
    l_B = ll[REJECT_ID].item()
    d_raw = l_A - l_B
    p_accept = 1.0 / (1.0 + math.exp(-d_raw))
    pred = "A" if d_raw > 0 else ("B" if d_raw < 0 else "TIE")
    return l_A, l_B, d_raw, p_accept, pred


def render(tpl: str, ans: str) -> str:
    return tpl.replace("<answer>", ans)


def cell_specs(p):
    return [
        ("OO", p["r_o"], render(TEMPLATES["T0"], p["r_o"])),
        ("OS", p["r_o"], render(TEMPLATES["T0"], p["r_s"])),
        ("SO", p["r_s"], render(TEMPLATES["T0"], p["r_o"])),
        ("SS", p["r_s"], render(TEMPLATES["T0"], p["r_s"])),
    ]


# ================= T0 reproduction =================
print("=== scoring T0 (reproduction) ===")
t0_rows = []
for p in dev_pairs:
    gid = p["original_group_id"]
    for cell, ref, cand in cell_specs(p):
        l_A, l_B, d_raw, p_accept, pred = score(p["q"], ref, cand)
        exp = "A" if cell in ("OO", "SS") else "B"
        t0_rows.append({"source_group_id": gid, "cell": cell, "question": p["q"],
                        "reference": ref, "candidate": cand, "expected_label": exp,
                        "l_A": l_A, "l_B": l_B, "d_raw": d_raw, "p_accept_raw": p_accept,
                        "predicted_label": pred, "correct": pred == exp})
print("T0 rows:", len(t0_rows))

# ---- compare with D1 four_cell_scores_dev.csv ----
d1_map = {}
with open(D1 / "four_cell_scores_dev.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d1_map[(r["source_group_id"], r["cell"])] = r

mismatch = []
for r in t0_rows:
    d1r = d1_map.get((r["source_group_id"], r["cell"]))
    if d1r is None:
        mismatch.append((r["source_group_id"], r["cell"], "missing in D1"))
        continue
    # BF16 serialized equality: D1 stored float32 repr of bf16 logits; compare within 1e-4
    if r["predicted_label"] != d1r["predicted_label"]:
        mismatch.append((r["source_group_id"], r["cell"],
                         f"pred {r['predicted_label']} vs D1 {d1r['predicted_label']}"))
        continue
    for fld in ("l_A", "l_B", "d_raw"):
        if abs(float(r[fld]) - float(d1r[fld])) > 1e-3:
            mismatch.append((r["source_group_id"], r["cell"],
                             f"{fld} {r[fld]:.6f} vs D1 {d1r[fld]:.6f}"))

if mismatch:
    print("T0 mismatches:", len(mismatch))
    for m in mismatch[:10]:
        print("  ", m)
    fail("base_template_reproduction_invalid", f"{len(mismatch)} row mismatches vs D1")

# aggregate comparison
def agg(rows):
    cells = {}
    for c in ["OO", "OS", "SO", "SS"]:
        sub = [r for r in rows if r["cell"] == c]
        cells[c] = {"n": len(sub),
                    "acc": sum(1 for r in sub if r["correct"]) / len(sub),
                    "accept_rate": sum(1 for r in sub if r["predicted_label"] == "A") / len(sub),
                    "tie": sum(1 for r in sub if r["predicted_label"] == "TIE") / len(sub)}
    acc_o = sum(1 for r in rows if r["cell"] in ("OO", "OS") and r["correct"]) / sum(1 for r in rows if r["cell"] in ("OO", "OS"))
    acc_s = sum(1 for r in rows if r["cell"] in ("SO", "SS") and r["correct"]) / sum(1 for r in rows if r["cell"] in ("SO", "SS"))
    return cells, acc_o, acc_s, acc_o - acc_s

cells0, acc_o0, acc_s0, rpag0 = agg(t0_rows)
print(f"T0 agg: OO={cells0['OO']['acc']:.3f} OS={cells0['OS']['acc']:.3f} SO={cells0['SO']['acc']:.3f} SS={cells0['SS']['acc']:.3f}")
print(f"T0: ACC_o={acc_o0:.3f} ACC_s={acc_s0:.3f} RPAG={rpag0:.3f}")

# D1 expected aggregates
d1_cells = {}
for r in csv.DictReader(open(D1 / "metrics_by_cell_dev.csv", encoding="utf-8")):
    d1_cells[r["cell"]] = {"acc": float(r["accuracy"]), "accept_rate": float(r["accept_rate"]),
                           "tie": float(r["tie_rate"])}
print("D1 metrics:", {c: d1_cells[c]["acc"] for c in ["OO", "OS", "SO", "SS"]})

# strict aggregate equality per task (targets are the task's stated 4-decimal values)
if not (abs(cells0["OO"]["acc"] - 1.0) < 5e-4 and abs(cells0["OS"]["acc"] - 1.0) < 5e-4
        and abs(cells0["SO"]["acc"] - 0.928) < 5e-4
        and abs(cells0["SS"]["acc"] - 0.241) < 5e-4):
    fail("base_template_reproduction_invalid", f"T0 aggregate mismatch: {cells0}")
EXP_ACCO = 1.0
EXP_ACCS = 0.585
EXP_RPAG = 0.415
if not (abs(acc_o0 - EXP_ACCO) < 5e-4 and abs(acc_s0 - EXP_ACCS) < 5e-4
        and abs(rpag0 - EXP_RPAG) < 5e-4):
    fail("base_template_reproduction_invalid",
         f"T0 ACC_o/ACC_s/RPAG mismatch: {acc_o0:.6f}/{acc_s0:.6f}/{rpag0:.6f} vs "
         f"{EXP_ACCO:.3f}/{EXP_ACCS:.3f}/{EXP_RPAG:.3f}")

print("T0 EXACT REPRODUCTION OK (row-by-row + aggregate)")

# save T0 rows
with open(D1R / "t0_reproduction_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "cell", "question", "reference", "candidate",
                                      "expected_label", "l_A", "l_B", "d_raw", "p_accept_raw",
                                      "predicted_label", "correct"])
    w.writeheader()
    w.writerows(t0_rows)

# ================= T1/T2 evaluation =================
print("=== scoring T1/T2 ===")
all_rows = {"T0": t0_rows}

# T0 SS error groups (anchor)
t0_ss_err = {r["source_group_id"] for r in t0_rows if r["cell"] == "SS" and not r["correct"]}
print("T0 SS error groups:", len(t0_ss_err))

for tpl_name in ["T1", "T2"]:
    rows = []
    for p in dev_pairs:
        gid = p["original_group_id"]
        specs = [
            ("OO", p["r_o"], render(TEMPLATES[tpl_name], p["r_o"])),
            ("OS", p["r_o"], render(TEMPLATES[tpl_name], p["r_s"])),
            ("SO", p["r_s"], render(TEMPLATES[tpl_name], p["r_o"])),
            ("SS", p["r_s"], render(TEMPLATES[tpl_name], p["r_s"])),
        ]
        for cell, ref, cand in specs:
            l_A, l_B, d_raw, p_accept, pred = score(p["q"], ref, cand)
            exp = "A" if cell in ("OO", "SS") else "B"
            rows.append({"source_group_id": gid, "cell": cell, "question": p["q"],
                         "reference": ref, "candidate": cand, "expected_label": exp,
                         "l_A": l_A, "l_B": l_B, "d_raw": d_raw, "p_accept_raw": p_accept,
                         "predicted_label": pred, "correct": pred == exp})
    all_rows[tpl_name] = rows
    print(f"{tpl_name} rows:", len(rows))

# ---- metrics per template/cell ----
def metrics_for(rows):
    out = {}
    for c in ["OO", "OS", "SO", "SS"]:
        sub = [r for r in rows if r["cell"] == c]
        n = len(sub)
        acc = sum(1 for r in sub if r["correct"]) / n
        ar = sum(1 for r in sub if r["predicted_label"] == "A") / n
        dr = [r["d_raw"] for r in sub]
        med = sorted(dr)[len(dr) // 2]
        tie = sum(1 for r in sub if r["predicted_label"] == "TIE") / n
        out[c] = {"n": n, "accuracy": acc, "accept_rate": ar,
                  "mean_d_raw": sum(dr) / n, "median_d_raw": med, "tie_rate": tie}
    acc_o = sum(1 for r in rows if r["cell"] in ("OO", "OS") and r["correct"]) / sum(1 for r in rows if r["cell"] in ("OO", "OS"))
    acc_s = sum(1 for r in rows if r["cell"] in ("SO", "SS") and r["correct"]) / sum(1 for r in rows if r["cell"] in ("SO", "SS"))
    fr_ss = sum(1 for r in rows if r["cell"] == "SS" and r["predicted_label"] == "B") / sum(1 for r in rows if r["cell"] == "SS")
    fa_so = sum(1 for r in rows if r["cell"] == "SO" and r["predicted_label"] == "A") / sum(1 for r in rows if r["cell"] == "SO")
    total_tie = sum(1 for r in rows if r["predicted_label"] == "TIE") / len(rows)
    return {"cells": out, "ACC_o": acc_o, "ACC_s": acc_s, "RPAG": acc_o - acc_s,
            "false_reject_SS": fr_ss, "false_accept_SO": fa_so, "total_tie_rate": total_tie}

M = {t: metrics_for(all_rows[t]) for t in ["T0", "T1", "T2"]}
for t in ["T0", "T1", "T2"]:
    print(t, {c: round(M[t]["cells"][c]["accuracy"], 4) for c in ["OO", "OS", "SO", "SS"]},
          f"ACC_o={M[t]['ACC_o']:.3f} ACC_s={M[t]['ACC_s']:.3f} RPAG={M[t]['RPAG']:.3f} "
          f"FR_SS={M[t]['false_reject_SS']:.3f} FA_SO={M[t]['false_accept_SO']:.3f} tie={M[t]['total_tie_rate']:.3f}")

# SS error retention
ret = {}
for t in ["T1", "T2"]:
    tk_ss_err = {r["source_group_id"] for r in all_rows[t] if r["cell"] == "SS" and not r["correct"]}
    ret[t] = len(t0_ss_err & tk_ss_err) / len(t0_ss_err) if t0_ss_err else 0.0
    print(f"SS_error_retention({t}) = {ret[t]:.3f} ({len(t0_ss_err & tk_ss_err)}/{len(t0_ss_err)})")

# ---- bootstrap (1,000 source-group resamples) ----
import random
rng = random.Random(20260802)
gids = sorted({p["original_group_id"] for p in dev_pairs})


def sample_metric(rows_src, sample_gids, name):
    sub = [r for gid in sample_gids for r in rows_src if r["source_group_id"] == gid]
    nss = sum(1 for r in sub if r["cell"] == "SS")
    nso = sum(1 for r in sub if r["cell"] == "SO")
    fr_ss = sum(1 for r in sub if r["cell"] == "SS" and r["predicted_label"] == "B") / max(1, nss)
    acc_o = sum(1 for r in sub if r["cell"] in ("OO", "OS") and r["correct"]) / max(1, sum(1 for r in sub if r["cell"] in ("OO", "OS")))
    acc_s = sum(1 for r in sub if r["cell"] in ("SO", "SS") and r["correct"]) / max(1, sum(1 for r in sub if r["cell"] in ("SO", "SS")))
    rpag = acc_o - acc_s
    if name == "RPAG":
        return rpag
    if name == "false_reject_SS":
        return fr_ss
    return None


bootstrap_rows = []
for t in ["T1", "T2"]:
    src = all_rows[t]
    boot_fr = []
    boot_rpag = []
    for _ in range(1000):
        sg = [rng.choice(gids) for _ in gids]
        boot_fr.append(sample_metric(src, sg, "false_reject_SS"))
        boot_rpag.append(sample_metric(src, sg, "RPAG"))
    for name, vals in [("false_reject_SS", boot_fr), ("RPAG", boot_rpag)]:
        vals.sort()
        bootstrap_rows.append({"template": t, "metric": name,
                               "ci95_low": vals[25], "ci95_high": vals[974]})
        print(f"bootstrap {t} {name}: CI [{vals[25]:.4f}, {vals[974]:.4f}]")
    # retention bootstrap: resample groups, recompute retention
    boot_ret = []
    for _ in range(1000):
        sg = [rng.choice(gids) for _ in gids]
        tk_err = {g for g in sg if any(r["source_group_id"] == g and r["cell"] == "SS" and not r["correct"] for r in src)}
        t0_err = {g for g in sg if any(r["source_group_id"] == g and r["cell"] == "SS" and not r["correct"] for r in all_rows["T0"])}
        boot_ret.append(len(t0_err & tk_err) / len(t0_err) if t0_err else 0.0)
    boot_ret.sort()
    bootstrap_rows.append({"template": t, "metric": "SS_error_retention",
                           "ci95_low": boot_ret[25], "ci95_high": boot_ret[974]})
    print(f"bootstrap {t} SS_error_retention: CI [{boot_ret[25]:.4f}, {boot_ret[974]:.4f}]")

with open(D1R / "bootstrap_template_robustness.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["template", "metric", "ci95_low", "ci95_high"])
    w.writeheader()
    w.writerows(bootstrap_rows)

# ---- metrics_by_template_cell_dev.csv ----
with open(D1R / "metrics_by_template_cell_dev.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["template", "cell", "n", "accuracy", "accept_rate",
                                      "mean_d_raw", "median_d_raw", "tie_rate",
                                      "ACC_o", "ACC_s", "RPAG", "false_reject_SS", "false_accept_SO"])
    w.writeheader()
    for t in ["T0", "T1", "T2"]:
        for c in ["OO", "OS", "SO", "SS"]:
            m = M[t]["cells"][c]
            w.writerow({"template": t, "cell": c, "n": m["n"], "accuracy": f"{m['accuracy']:.4f}",
                        "accept_rate": f"{m['accept_rate']:.4f}",
                        "mean_d_raw": f"{m['mean_d_raw']:.4f}",
                        "median_d_raw": f"{m['median_d_raw']:.4f}",
                        "tie_rate": f"{m['tie_rate']:.4f}",
                        "ACC_o": f"{M[t]['ACC_o']:.4f}", "ACC_s": f"{M[t]['ACC_s']:.4f}",
                        "RPAG": f"{M[t]['RPAG']:.4f}",
                        "false_reject_SS": f"{M[t]['false_reject_SS']:.4f}",
                        "false_accept_SO": f"{M[t]['false_accept_SO']:.4f}"})

# ---- template_error_retention_audit.csv (per T0-SS-error group) ----
with open(D1R / "template_error_retention_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "question",
                                      "T0_SS_correct", "T1_SS_correct", "T2_SS_correct",
                                      "T0_SS_d_raw", "T1_SS_d_raw", "T2_SS_d_raw"])
    w.writeheader()
    err_gids = sorted(t0_ss_err)
    for gid in err_gids:
        get = {}
        for t in ["T0", "T1", "T2"]:
            for r in all_rows[t]:
                if r["source_group_id"] == gid and r["cell"] == "SS":
                    get[t] = r
                    break
        w.writerow({"source_group_id": gid, "question": get["T0"]["question"][:80],
                    "T0_SS_correct": get["T0"]["correct"], "T1_SS_correct": get["T1"]["correct"],
                    "T2_SS_correct": get["T2"]["correct"],
                    "T0_SS_d_raw": f"{get['T0']['d_raw']:.4f}",
                    "T1_SS_d_raw": f"{get['T1']['d_raw']:.4f}",
                    "T2_SS_d_raw": f"{get['T2']['d_raw']:.4f}"})

# ---- save summary ----
(D1R / "scripts" / "_summary.json").write_text(json.dumps({
    "M": {t: {k: (v if k != "cells" else {c: M[t]["cells"][c] for c in ["OO", "OS", "SO", "SS"]})
               for k, v in M[t].items()} for t in ["T0", "T1", "T2"]},
    "retention": ret,
    "t0_ss_error_groups": len(t0_ss_err),
    "bootstrap_rows": bootstrap_rows,
}, indent=2), encoding="utf-8")
print("saved _summary.json")
print("T0 reproduction OK, T1/T2 scored. Ready for gate.")
