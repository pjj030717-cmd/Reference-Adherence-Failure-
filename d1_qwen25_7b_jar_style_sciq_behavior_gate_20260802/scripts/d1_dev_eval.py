#!/usr/bin/env python3
"""E01-D1: dev four-cell behavior evaluation.

Runs teacher-forced scoring on all four cells of each of the 195 dev source groups,
plus the no-reference greedy knowledge diagnostic, and computes gate metrics
(ACC_o, ACC_s, RPAG, false_reject_SS, false_accept_SO) with 1000 bootstrap CIs.

No hidden states are extracted; only final-layer logits for continuation tokens.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
import unicodedata
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL, ACCEPT, REJECT = CONST["system"], CONST["user_template"], CONST["accept"], CONST["reject"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]

dev_pairs = []
with open(D1 / "scripts" / "_dev_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dev_pairs.append(json.loads(line))
print("dev pairs loaded:", len(dev_pairs))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()
print("model loaded for dev evaluation")


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


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ---- four-cell evaluation ----
rows = []
for p in dev_pairs:
    gid = p["original_group_id"]
    q = p["q"]
    r_o, r_s = p["r_o"], p["r_s"]
    c_o, c_s = p["c_o"], p["c_s"]
    cells = [
        ("OO", r_o, c_o, "A"),
        ("OS", r_o, c_s, "B"),
        ("SO", r_s, c_o, "B"),
        ("SS", r_s, c_s, "A"),
    ]
    for cell, ref, cand, exp in cells:
        l_A, l_B, d_raw, p_accept, pred = score(q, ref, cand)
        rows.append({"source_group_id": gid, "cell": cell, "question": q,
                     "reference": ref, "candidate": cand, "expected_label": exp,
                     "l_A": l_A, "l_B": l_B, "d_raw": d_raw, "p_accept_raw": p_accept,
                     "predicted_label": pred, "correct": pred == exp})
        if cell == "OO":
            pass

print("four-cell rows scored:", len(rows))

# ---- metrics by cell ----
cells = ["OO", "OS", "SO", "SS"]
metrics = {}
for c in cells:
    sub = [r for r in rows if r["cell"] == c]
    n = len(sub)
    acc = sum(1 for r in sub if r["correct"]) / n
    accept_rate = sum(1 for r in sub if r["predicted_label"] == "A") / n
    dr = [r["d_raw"] for r in sub]
    med = sorted(dr)[len(dr) // 2]
    tie = sum(1 for r in sub if r["predicted_label"] == "TIE") / n
    metrics[c] = {"n": n, "accuracy": acc, "accept_rate": accept_rate,
                  "mean_d_raw": sum(dr) / len(dr), "median_d_raw": med, "tie_rate": tie}
    print(c, metrics[c])

# ACC_o, ACC_s
ACC_o = (sum(1 for r in rows if r["cell"] in ("OO", "OS") and r["correct"])
         / sum(1 for r in rows if r["cell"] in ("OO", "OS")))
ACC_s = (sum(1 for r in rows if r["cell"] in ("SO", "SS") and r["correct"])
         / sum(1 for r in rows if r["cell"] in ("SO", "SS")))
RPAG = ACC_o - ACC_s
fr_SS = sum(1 for r in rows if r["cell"] == "SS" and r["predicted_label"] == "B") / metrics["SS"]["n"]
fa_SO = sum(1 for r in rows if r["cell"] == "SO" and r["predicted_label"] == "A") / metrics["SO"]["n"]
print(f"ACC_o={ACC_o:.4f} ACC_s={ACC_s:.4f} RPAG={RPAG:.4f} FR_SS={fr_SS:.4f} FA_SO={fa_SO:.4f}")

# override_error_group: group with >=1 error in SO or SS
by_group = {}
for r in rows:
    by_group.setdefault(r["source_group_id"], {}).setdefault(r["cell"], []).append(r)
override_err_groups = []
for gid, cmap in by_group.items():
    for c in ("SO", "SS"):
        if any(not r["correct"] for r in cmap.get(c, [])):
            override_err_groups.append(gid)
            break
print("override_error_groups:", len(override_err_groups))

# ---- 1000 bootstrap (source-group level) ----
rng = random.Random(20260802)
gids = list(by_group.keys())


def sample_metrics(sample_gids):
    sub = [r for gid in sample_gids for r in rows if r["source_group_id"] == gid]
    acc_o = sum(1 for r in sub if r["cell"] in ("OO", "OS") and r["correct"]) / max(1, sum(1 for r in sub if r["cell"] in ("OO", "OS")))
    acc_s = sum(1 for r in sub if r["cell"] in ("SO", "SS") and r["correct"]) / max(1, sum(1 for r in sub if r["cell"] in ("SO", "SS")))
    nss = sum(1 for r in sub if r["cell"] == "SS")
    nso = sum(1 for r in sub if r["cell"] == "SO")
    fr_ss = sum(1 for r in sub if r["cell"] == "SS" and r["predicted_label"] == "B") / max(1, nss)
    fa_so = sum(1 for r in sub if r["cell"] == "SO" and r["predicted_label"] == "A") / max(1, nso)
    return acc_o, acc_s, acc_o - acc_s, fr_ss, fa_so


boot = [sample_metrics([rng.choice(gids) for _ in gids]) for _ in range(1000)]
boot_ci = {}
for i, name in enumerate(["ACC_o", "ACC_s", "RPAG", "false_reject_SS", "false_accept_SO"]):
    vals = sorted(b[i] for b in boot)
    boot_ci[name] = {"ci95_low": vals[25], "ci95_high": vals[974]}
    print(f"bootstrap {name}: 95% CI [{vals[25]:.4f}, {vals[974]:.4f}]")

# ---- write metrics_by_cell_dev.csv ----
with open(D1 / "metrics_by_cell_dev.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["cell", "n", "accuracy", "accept_rate", "mean_d_raw", "median_d_raw", "tie_rate",
                                      "ACC_o", "ACC_s", "RPAG", "false_reject_SS", "false_accept_SO"])
    w.writeheader()
    for c in cells:
        m = metrics[c]
        w.writerow({"cell": c, "n": m["n"], "accuracy": f"{m['accuracy']:.4f}",
                    "accept_rate": f"{m['accept_rate']:.4f}", "mean_d_raw": f"{m['mean_d_raw']:.4f}",
                    "median_d_raw": f"{m['median_d_raw']:.4f}", "tie_rate": f"{m['tie_rate']:.4f}",
                    "ACC_o": f"{ACC_o:.4f}", "ACC_s": f"{ACC_s:.4f}", "RPAG": f"{RPAG:.4f}",
                    "false_reject_SS": f"{fr_SS:.4f}", "false_accept_SO": f"{fa_SO:.4f}"})

# ---- counterfactual_group_audit.csv (per group) ----
with open(D1 / "counterfactual_group_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "question", "r_o", "r_s",
                                      "OO_correct", "OS_correct", "SO_correct", "SS_correct",
                                      "OO_d_raw", "OS_d_raw", "SO_d_raw", "SS_d_raw"])
    w.writeheader()
    for gid in gids:
        cmap = by_group[gid]
        d = {c: [r for r in cmap.get(c, [])][0] for c in cells}
        w.writerow({"source_group_id": gid, "question": d["OO"]["question"],
                    "r_o": d["OO"]["reference"], "r_s": d["SO"]["reference"],
                    "OO_correct": d["OO"]["correct"], "OS_correct": d["OS"]["correct"],
                    "SO_correct": d["SO"]["correct"], "SS_correct": d["SS"]["correct"],
                    "OO_d_raw": f"{d['OO']['d_raw']:.4f}", "OS_d_raw": f"{d['OS']['d_raw']:.4f}",
                    "SO_d_raw": f"{d['SO']['d_raw']:.4f}", "SS_d_raw": f"{d['SS']['d_raw']:.4f}"})

# ---- full per-row CSV (four cells) ----
with open(D1 / "scripts" / "_dev_fourcell_rows.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "cell", "question", "reference", "candidate",
                                      "expected_label", "l_A", "l_B", "d_raw", "p_accept_raw",
                                      "predicted_label", "correct"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print("wrote metrics_by_cell_dev.csv, counterfactual_group_audit.csv, _dev_fourcell_rows.csv")

# ---- parameter-knowledge diagnostic (no reference, greedy short answer) ----
KNOW_TMPL = "Answer the following question with a short answer only.\n\nQuestion: {question}\n\nAnswer:"


def norm_out(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    s = re.sub(r"[.!?\s]+$", "", s)
    return s.strip()


def r_o_match(out: str, r_o: str) -> bool:
    on, rn = norm_out(out), norm_out(r_o)
    if on == rn:
        return True
    # output starts with r_o followed only by punctuation/whitespace
    for suf in ("", ".", "!", "?"):
        if on.startswith(rn + suf):
            return True
    return False


know_rows = []
with torch.inference_mode():
    for p in dev_pairs:
        messages = [{"role": "user", "content": KNOW_TMPL.format(question=p["q"])}]
        enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        prompt_ids = enc["input_ids"].to("cuda")
        out_ids = model.generate(prompt_ids, max_new_tokens=20, do_sample=False, temperature=0.0)
        new_ids = out_ids[0, prompt_ids.shape[1]:]
        out_text = tok.decode(new_ids, skip_special_tokens=True)
        match = r_o_match(out_text, p["r_o"])
        know_rows.append({"source_group_id": p["original_group_id"], "question": p["q"],
                          "r_o": p["r_o"], "greedy_answer": out_text, "knowledge_matches_original": match})

km = sum(1 for r in know_rows if r["knowledge_matches_original"])
print(f"knowledge_matches_original: {km}/{len(know_rows)} = {km/len(know_rows):.3f}")

with open(D1 / "raw_knowledge_diagnostic_dev.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "question", "r_o", "greedy_answer", "knowledge_matches_original"])
    w.writeheader()
    w.writerows(know_rows)

# save summaries for gate script
(D1 / "scripts" / "_dev_summary.json").write_text(json.dumps({
    "ACC_o": ACC_o, "ACC_s": ACC_s, "RPAG": RPAG,
    "false_reject_SS": fr_SS, "false_accept_SO": fa_SO,
    "override_error_groups": override_err_groups,
    "n_override_err_groups": len(override_err_groups),
    "metrics": metrics,
    "bootstrap_ci": boot_ci,
    "total_tie_rate": sum(1 for r in rows if r["predicted_label"] == "TIE") / len(rows),
    "total_rows": len(rows),
    "knowledge_match_count": km, "knowledge_match_rate": km / len(know_rows),
    "cells_rows": {c: metrics[c] for c in cells},
}, indent=2), encoding="utf-8")
print("saved _dev_summary.json")
