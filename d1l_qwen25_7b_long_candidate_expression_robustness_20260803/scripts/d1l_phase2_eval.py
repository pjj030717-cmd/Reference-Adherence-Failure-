#!/usr/bin/env python3
"""E01-D1-L Phase 2: T3/T4/T5 four-cell behavior evaluation + bootstrap.

For each of T3-bare / T4-long-first / T5-long-last:
  - 195 groups x 4 cells = 780 fixed A/B judgments (teacher-forced, pos = prompt_len-1)
  - report OO/OS/SO/SS acc, ACC_o, ACC_s, RPAG, FR_SS, FA_SO, tie rate, d_raw stats
  - retention(T): within T0's 148 SS-false-reject groups, fraction still false-rejected under T
  - bootstrap 2000 (seed=20260818) source-group resamples: 95% CI for FR_SS, RPAG, retention

Writes: metrics_by_template_cell_dev.csv, ss_error_retention_audit.csv, bootstrap_behavior_metrics.csv.
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
OUT = REPO_ROOT / "d1l_qwen25_7b_long_candidate_expression_robustness_20260803"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"

CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]

spec = json.loads((OUT / "candidate_length_expression_spec.json").read_text(encoding="utf-8"))
T0 = spec["templates"]["T0"]["template"]
TPL = {n: spec["templates"][n]["template"] for n in ("T3", "T4", "T5")}

dev_pairs = []
with open(OUT / "scripts" / "_dev_input.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dev_pairs.append(json.loads(line))
t0_ss_err = set(json.loads((OUT / "scripts" / "_t0_ss_error_groups.json").read_text(encoding="utf-8")))
print("dev pairs:", len(dev_pairs), "| T0 SS error groups:", len(t0_ss_err))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()
print("model loaded for Phase 2")


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


def render(tpl, ans):
    return tpl.replace("<answer>", ans)


all_rows = {}
for name in ("T3", "T4", "T5"):
    tpl = TPL[name]
    rows = []
    for p in dev_pairs:
        gid = p["original_group_id"]
        specs = [
            ("OO", p["r_o"], render(tpl, p["r_o"])),
            ("OS", p["r_o"], render(tpl, p["r_s"])),
            ("SO", p["r_s"], render(tpl, p["r_o"])),
            ("SS", p["r_s"], render(tpl, p["r_s"])),
        ]
        for cell, ref, cand in specs:
            l_A, l_B, d_raw, p_accept, pred = score(p["q"], ref, cand)
            exp = "A" if cell in ("OO", "SS") else "B"
            rows.append({"source_group_id": gid, "cell": cell, "question": p["q"],
                         "reference": ref, "candidate": cand, "expected_label": exp,
                         "l_A": l_A, "l_B": l_B, "d_raw": d_raw, "p_accept_raw": p_accept,
                         "predicted_label": pred, "correct": pred == exp})
    all_rows[name] = rows
    print(f"{name} scored:", len(rows))


def metrics_for(rows):
    cells = {}
    for c in ["OO", "OS", "SO", "SS"]:
        sub = [r for r in rows if r["cell"] == c]
        n = len(sub)
        dr = [r["d_raw"] for r in sub]
        sdr = sorted(dr)
        q = lambda x: sdr[min(len(sdr) - 1, int(x * (len(sdr) - 1)))]
        cells[c] = {"n": n, "accuracy": sum(1 for r in sub if r["correct"]) / n,
                    "accept_rate": sum(1 for r in sub if r["predicted_label"] == "A") / n,
                    "mean_d_raw": sum(dr) / n, "median_d_raw": sdr[n // 2],
                    "q25": q(0.25), "q75": q(0.75), "tie_rate": sum(1 for r in sub if r["predicted_label"] == "TIE") / n}
    acc_o = sum(1 for r in rows if r["cell"] in ("OO", "OS") and r["correct"]) / sum(1 for r in rows if r["cell"] in ("OO", "OS"))
    acc_s = sum(1 for r in rows if r["cell"] in ("SO", "SS") and r["correct"]) / sum(1 for r in rows if r["cell"] in ("SO", "SS"))
    fr_ss = sum(1 for r in rows if r["cell"] == "SS" and r["predicted_label"] == "B") / sum(1 for r in rows if r["cell"] == "SS")
    fa_so = sum(1 for r in rows if r["cell"] == "SO" and r["predicted_label"] == "A") / sum(1 for r in rows if r["cell"] == "SO")
    total_tie = sum(1 for r in rows if r["predicted_label"] == "TIE") / len(rows)
    return {"cells": cells, "ACC_o": acc_o, "ACC_s": acc_s, "RPAG": acc_o - acc_s,
            "false_reject_SS": fr_ss, "false_accept_SO": fa_so, "total_tie_rate": total_tie}


M = {n: metrics_for(all_rows[n]) for n in ("T3", "T4", "T5")}
for n in ("T3", "T4", "T5"):
    m = M[n]
    print(n, {c: round(m["cells"][c]["accuracy"], 4) for c in ["OO", "OS", "SO", "SS"]},
          f"ACC_o={m['ACC_o']:.3f} ACC_s={m['ACC_s']:.3f} RPAG={m['RPAG']:.3f} "
          f"FR_SS={m['false_reject_SS']:.3f} FA_SO={m['false_accept_SO']:.3f} tie={m['total_tie_rate']:.3f}")

# ---- retention ----
ret = {}
for n in ("T3", "T4", "T5"):
    err_n = {r["source_group_id"] for r in all_rows[n] if r["cell"] == "SS" and r["predicted_label"] == "B"}
    ret[n] = len(t0_ss_err & err_n) / len(t0_ss_err)
    print(f"retention({n}) = {ret[n]:.4f} ({len(t0_ss_err & err_n)}/{len(t0_ss_err)})")

# ---- bootstrap 2000, seed 20260818, source-group resamples ----
rng = random.Random(20260818)
gids = sorted({p["original_group_id"] for p in dev_pairs})


def sample_metrics(rows_src, sample_gids):
    sub = [r for gid in sample_gids for r in rows_src if r["source_group_id"] == gid]
    acc_o = sum(1 for r in sub if r["cell"] in ("OO", "OS") and r["correct"]) / max(1, sum(1 for r in sub if r["cell"] in ("OO", "OS")))
    acc_s = sum(1 for r in sub if r["cell"] in ("SO", "SS") and r["correct"]) / max(1, sum(1 for r in sub if r["cell"] in ("SO", "SS")))
    fr_ss = sum(1 for r in sub if r["cell"] == "SS" and r["predicted_label"] == "B") / max(1, sum(1 for r in sub if r["cell"] == "SS"))
    return acc_o - acc_s, fr_ss


def sample_retention(rows_src, sample_gids):
    t0_err = {g for g in sample_gids if g in t0_ss_err}
    t_err = {g for g in sample_gids if g in t0_ss_err
             and any(r["source_group_id"] == g and r["cell"] == "SS" and r["predicted_label"] == "B" for r in rows_src)}
    return len(t0_err & t_err) / len(t0_err) if t0_err else 0.0


bootstrap_rows = []
for n in ("T3", "T4", "T5"):
    src = all_rows[n]
    boot_rpag = []
    boot_fr = []
    boot_ret = []
    for _ in range(2000):
        sg = [rng.choice(gids) for _ in gids]
        rpag, fr = sample_metrics(src, sg)
        boot_rpag.append(rpag)
        boot_fr.append(fr)
        boot_ret.append(sample_retention(src, sg))
    for metric, vals in [("false_reject_SS", boot_fr), ("RPAG", boot_rpag), ("retention", boot_ret)]:
        vals.sort()
        lo, hi = vals[50], vals[1949]
        bootstrap_rows.append({"template": n, "metric": metric, "ci95_low": lo, "ci95_high": hi})
        print(f"bootstrap {n} {metric}: CI [{lo:.4f}, {hi:.4f}]")

with open(OUT / "bootstrap_behavior_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["template", "metric", "ci95_low", "ci95_high"])
    w.writeheader()
    w.writerows(bootstrap_rows)

# ---- metrics_by_template_cell_dev.csv ----
with open(OUT / "metrics_by_template_cell_dev.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["template", "cell", "n", "accuracy", "accept_rate",
                                      "mean_d_raw", "median_d_raw", "q25_d_raw", "q75_d_raw", "tie_rate",
                                      "ACC_o", "ACC_s", "RPAG", "false_reject_SS", "false_accept_SO",
                                      "SS_error_retention"])
    w.writeheader()
    for n in ("T3", "T4", "T5"):
        for c in ["OO", "OS", "SO", "SS"]:
            m = M[n]["cells"][c]
            w.writerow({"template": n, "cell": c, "n": m["n"], "accuracy": f"{m['accuracy']:.4f}",
                        "accept_rate": f"{m['accept_rate']:.4f}",
                        "mean_d_raw": f"{m['mean_d_raw']:.4f}", "median_d_raw": f"{m['median_d_raw']:.4f}",
                        "q25_d_raw": f"{m['q25']:.4f}", "q75_d_raw": f"{m['q75']:.4f}",
                        "tie_rate": f"{m['tie_rate']:.4f}",
                        "ACC_o": f"{M[n]['ACC_o']:.4f}", "ACC_s": f"{M[n]['ACC_s']:.4f}",
                        "RPAG": f"{M[n]['RPAG']:.4f}",
                        "false_reject_SS": f"{M[n]['false_reject_SS']:.4f}",
                        "false_accept_SO": f"{M[n]['false_accept_SO']:.4f}",
                        "SS_error_retention": f"{ret[n]:.4f}"})

# ---- ss_error_retention_audit.csv (per T0 SS error group) ----
with open(OUT / "ss_error_retention_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "question",
                                      "T3_SS_false_reject", "T4_SS_false_reject", "T5_SS_false_reject",
                                      "T3_SS_d_raw", "T4_SS_d_raw", "T5_SS_d_raw"])
    w.writeheader()
    for gid in sorted(t0_ss_err):
        q = ""
        vals = {}
        for n in ("T3", "T4", "T5"):
            for r in all_rows[n]:
                if r["source_group_id"] == gid and r["cell"] == "SS":
                    vals[n] = r
                    q = r["question"]
                    break
        w.writerow({"source_group_id": gid, "question": q[:80],
                    "T3_SS_false_reject": vals["T3"]["predicted_label"] == "B",
                    "T4_SS_false_reject": vals["T4"]["predicted_label"] == "B",
                    "T5_SS_false_reject": vals["T5"]["predicted_label"] == "B",
                    "T3_SS_d_raw": f"{vals['T3']['d_raw']:.4f}",
                    "T4_SS_d_raw": f"{vals['T4']['d_raw']:.4f}",
                    "T5_SS_d_raw": f"{vals['T5']['d_raw']:.4f}"})

# save summary for gate
(OUT / "scripts" / "_phase2_summary.json").write_text(json.dumps({
    "M": {n: {k: (v if k != "cells" else {c: M[n]["cells"][c] for c in ["OO", "OS", "SO", "SS"]})
               for k, v in M[n].items()} for n in ("T3", "T4", "T5")},
    "retention": ret,
    "t0_ss_error_groups": len(t0_ss_err),
    "bootstrap": bootstrap_rows,
}, indent=2), encoding="utf-8")

# save per-row data for all templates (audit)
with open(OUT / "scripts" / "_t345_fourcell_rows.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["template", "source_group_id", "cell", "question", "reference",
                                      "candidate", "expected_label", "l_A", "l_B", "d_raw",
                                      "p_accept_raw", "predicted_label", "correct"])
    w.writeheader()
    for n in ("T3", "T4", "T5"):
        for r in all_rows[n]:
            w.writerow({"template": n, **{k: r[k] for k in r if k != "template"}})
print("saved _t345_fourcell_rows.csv")
print("Phase 2 OK")
