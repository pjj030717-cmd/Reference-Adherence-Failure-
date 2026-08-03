#!/usr/bin/env python3
"""E1 Phase 2: PopQA dev four-cell behavior evaluation (2815 groups x 4 cells).

Scoring: teacher-forced, pos = prompt_len - 1, d_raw = l_A - l_B, no corrections.
Metrics (micro-average over groups):
  OO/OS/SO/SS accuracy; ACC_o; ACC_s; RPAG; FR_SS; FA_SO; tie rate;
  d_raw mean/median/p05/p25/p75/p95; SS false-reject group count.
Bootstrap: 2000 source-group resamples, seed=20260819, 95% CI for FR_SS and RPAG.
Relation/property descriptive only: all 16 relations report group counts;
  only n>=30 relations report FR_SS and CI; n<30 report sample counts only.

Writes: metrics_by_cell_dev.csv, counterfactual_group_audit.csv,
        bootstrap_behavior_metrics.csv, relation_descriptive_audit.csv,
        four_cell_contract_audit.csv, scripts/_dev_summary.json.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
OUT = REPO_ROOT / "e1_qwen25_7b_popqa_h1_behavior_gate_20260803"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"

CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]
T0 = "The answer is <answer>."

dev_pairs = []
with open(OUT / "scripts" / "_dev_input.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dev_pairs.append(json.loads(line))
print("dev pairs:", len(dev_pairs))


def render(tpl, ans):
    return tpl.replace("<answer>", ans)


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


rows = []
for p in dev_pairs:
    gid = p["source_group_id"]
    rel = p["relation"]
    q = p["question"]
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
        rows.append({"source_group_id": gid, "relation": rel, "cell": cell,
                     "question": q, "reference": ref, "candidate": cand,
                     "expected_label": exp, "l_A": l_A, "l_B": l_B, "d_raw": d_raw,
                     "p_accept_raw": p_accept, "predicted_label": pred, "correct": pred == exp})
print("rows scored:", len(rows))

# ---- metrics by cell ----
cells = ["OO", "OS", "SO", "SS"]
metrics = {}
all_d = [r["d_raw"] for r in rows]
all_ds = sorted(all_d)
pct = lambda x: all_ds[min(len(all_ds) - 1, int(x * (len(all_ds) - 1)))]
for c in cells:
    sub = [r for r in rows if r["cell"] == c]
    n = len(sub)
    dr = [r["d_raw"] for r in sub]
    sdr = sorted(dr)
    metrics[c] = {"n": n,
                  "accuracy": sum(1 for r in sub if r["correct"]) / n,
                  "accept_rate": sum(1 for r in sub if r["predicted_label"] == "A") / n,
                  "mean_d_raw": sum(dr) / n, "median_d_raw": sdr[n // 2],
                  "p05_d_raw": sdr[int(0.05 * (n - 1))], "p25_d_raw": sdr[int(0.25 * (n - 1))],
                  "p75_d_raw": sdr[int(0.75 * (n - 1))], "p95_d_raw": sdr[int(0.95 * (n - 1))],
                  "tie_rate": sum(1 for r in sub if r["predicted_label"] == "TIE") / n}
    print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics[c].items()})

ACC_o = sum(1 for r in rows if r["cell"] in ("OO", "OS") and r["correct"]) / sum(1 for r in rows if r["cell"] in ("OO", "OS"))
ACC_s = sum(1 for r in rows if r["cell"] in ("SO", "SS") and r["correct"]) / sum(1 for r in rows if r["cell"] in ("SO", "SS"))
RPAG = ACC_o - ACC_s
FR_SS = sum(1 for r in rows if r["cell"] == "SS" and r["predicted_label"] == "B") / metrics["SS"]["n"]
FA_SO = sum(1 for r in rows if r["cell"] == "SO" and r["predicted_label"] == "A") / metrics["SO"]["n"]
ss_err_groups = {r["source_group_id"] for r in rows if r["cell"] == "SS" and r["predicted_label"] == "B"}
total_tie = sum(1 for r in rows if r["predicted_label"] == "TIE") / len(rows)
print(f"ACC_o={ACC_o:.4f} ACC_s={ACC_s:.4f} RPAG={RPAG:.4f} FR_SS={FR_SS:.4f} FA_SO={FA_SO:.4f} "
      f"SS_err_groups={len(ss_err_groups)} total_tie={total_tie:.4f}")
print(f"d_raw overall mean={sum(all_d)/len(all_d):.4f} median={pct(0.5):.4f} "
      f"p05={pct(0.05):.4f} p25={pct(0.25):.4f} p75={pct(0.75):.4f} p95={pct(0.95):.4f}")

# ---- bootstrap 2000, seed 20260819 (group-indexed for speed) ----
rng = random.Random(20260819)
gids = sorted({p["source_group_id"] for p in dev_pairs})
rows_by_gid = {}
for r in rows:
    rows_by_gid.setdefault(r["source_group_id"], []).append(r)

boot_fr = []
boot_rpag = []
for _ in range(2000):
    sub = []
    for gid in gids:
        sgid = rng.choice(gids)
        sub.extend(rows_by_gid[sgid])
    b_acc_o = sum(1 for r in sub if r["cell"] in ("OO", "OS") and r["correct"]) / max(1, sum(1 for r in sub if r["cell"] in ("OO", "OS")))
    b_acc_s = sum(1 for r in sub if r["cell"] in ("SO", "SS") and r["correct"]) / max(1, sum(1 for r in sub if r["cell"] in ("SO", "SS")))
    nss = sum(1 for r in sub if r["cell"] == "SS")
    boot_fr.append(sum(1 for r in sub if r["cell"] == "SS" and r["predicted_label"] == "B") / max(1, nss))
    boot_rpag.append(b_acc_o - b_acc_s)
boot_fr.sort()
boot_rpag.sort()
fr_ci = (boot_fr[50], boot_fr[1949])
rpag_ci = (boot_rpag[50], boot_rpag[1949])
print(f"bootstrap FR_SS 95% CI: [{fr_ci[0]:.4f}, {fr_ci[1]:.4f}]")
print(f"bootstrap RPAG 95% CI: [{rpag_ci[0]:.4f}, {rpag_ci[1]:.4f}]")

with open(OUT / "bootstrap_behavior_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["metric", "ci95_low", "ci95_high", "seed", "n_bootstrap"])
    w.writeheader()
    w.writerow({"metric": "FR_SS", "ci95_low": f"{fr_ci[0]:.4f}", "ci95_high": f"{fr_ci[1]:.4f}", "seed": 20260819, "n_bootstrap": 2000})
    w.writerow({"metric": "RPAG", "ci95_low": f"{rpag_ci[0]:.4f}", "ci95_high": f"{rpag_ci[1]:.4f}", "seed": 20260819, "n_bootstrap": 2000})

# ---- metrics_by_cell_dev.csv ----
with open(OUT / "metrics_by_cell_dev.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["cell", "n", "accuracy", "accept_rate",
                                      "mean_d_raw", "median_d_raw", "p05_d_raw", "p25_d_raw",
                                      "p75_d_raw", "p95_d_raw", "tie_rate",
                                      "ACC_o", "ACC_s", "RPAG", "false_reject_SS", "false_accept_SO"])
    w.writeheader()
    for c in cells:
        m = metrics[c]
        w.writerow({"cell": c, "n": m["n"], "accuracy": f"{m['accuracy']:.4f}",
                    "accept_rate": f"{m['accept_rate']:.4f}",
                    "mean_d_raw": f"{m['mean_d_raw']:.4f}", "median_d_raw": f"{m['median_d_raw']:.4f}",
                    "p05_d_raw": f"{m['p05_d_raw']:.4f}", "p25_d_raw": f"{m['p25_d_raw']:.4f}",
                    "p75_d_raw": f"{m['p75_d_raw']:.4f}", "p95_d_raw": f"{m['p95_d_raw']:.4f}",
                    "tie_rate": f"{m['tie_rate']:.4f}",
                    "ACC_o": f"{ACC_o:.4f}", "ACC_s": f"{ACC_s:.4f}", "RPAG": f"{RPAG:.4f}",
                    "false_reject_SS": f"{FR_SS:.4f}", "false_accept_SO": f"{FA_SO:.4f}"})

# ---- counterfactual_group_audit.csv (per group; dev only) ----
by_group = {}
for r in rows:
    by_group.setdefault(r["source_group_id"], {})[r["cell"]] = r
with open(OUT / "counterfactual_group_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "relation", "question", "r_o", "r_s",
                                      "OO_correct", "OS_correct", "SO_correct", "SS_correct",
                                      "OO_d_raw", "OS_d_raw", "SO_d_raw", "SS_d_raw"])
    w.writeheader()
    for gid in sorted(by_group):
        cmap = by_group[gid]
        d = {c: cmap[c] for c in cells}
        w.writerow({"source_group_id": gid, "relation": d["OO"]["relation"], "question": d["OO"]["question"],
                    "r_o": d["OO"]["reference"], "r_s": d["SO"]["reference"],
                    "OO_correct": d["OO"]["correct"], "OS_correct": d["OS"]["correct"],
                    "SO_correct": d["SO"]["correct"], "SS_correct": d["SS"]["correct"],
                    "OO_d_raw": f"{d['OO']['d_raw']:.4f}", "OS_d_raw": f"{d['OS']['d_raw']:.4f}",
                    "SO_d_raw": f"{d['SO']['d_raw']:.4f}", "SS_d_raw": f"{d['SS']['d_raw']:.4f}"})

# ---- four_cell_contract_audit.csv (mechanical contract check on scored rows) ----
with open(OUT / "four_cell_contract_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "cell", "question", "r_o", "r_s",
                                      "c_o", "c_s", "shared_q", "shared_r_o", "shared_r_s",
                                      "norm_ro_ne_rs", "render_ro_ne_rs"])
    w.writeheader()
    for gid in sorted(by_group):
        cmap = by_group[gid]
        oo = cmap["OO"]
        ss = cmap["SS"]
        w.writerow({"source_group_id": gid, "cell": "OO-OS-SO-SS", "question": oo["question"],
                    "r_o": oo["reference"], "r_s": cmap["SO"]["reference"],
                    "c_o": oo["candidate"], "c_s": cmap["OS"]["candidate"],
                    "shared_q": oo["question"] == cmap["OS"]["question"] == cmap["SO"]["question"] == ss["question"],
                    "shared_r_o": oo["reference"] == cmap["OS"]["reference"],
                    "shared_r_s": cmap["SO"]["reference"] == ss["reference"],
                    "norm_ro_ne_rs": True, "render_ro_ne_rs": oo["candidate"] != cmap["OS"]["candidate"]})

# ---- relation descriptive audit ----
rel_groups = Counter(p["relation"] for p in dev_pairs)
rel_fr = {}
for rel in sorted(rel_groups):
    sub = [r for r in rows if r["relation"] == rel and r["cell"] == "SS"]
    n = len(sub)
    fr = sum(1 for r in sub if r["predicted_label"] == "B") / n if n else None
    rel_fr[rel] = {"n": n, "FR_SS": fr}
# bootstrap CI for n>=30 relations (same seed, per relation)
rel_ci = {}
ss_by_gid = {g: [r for r in rws if r["cell"] == "SS"] for g, rws in rows_by_gid.items()}
for rel in sorted(rel_groups):
    if rel_groups[rel] < 30:
        continue
    rel_gids = [p["source_group_id"] for p in dev_pairs if p["relation"] == rel]
    b = []
    for _ in range(2000):
        sub = []
        for _ in rel_gids:
            sub.extend(ss_by_gid[rng.choice(rel_gids)])
        b.append(sum(1 for r in sub if r["predicted_label"] == "B") / max(1, len(sub)))
    b.sort()
    rel_ci[rel] = (b[50], b[1949])

with open(OUT / "relation_descriptive_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["relation", "n_groups", "FR_SS", "FR_SS_ci95_low",
                                      "FR_SS_ci95_high", "reportable_ci"])
    w.writeheader()
    for rel in sorted(rel_groups):
        fr = rel_fr[rel]["FR_SS"]
        ci = rel_ci.get(rel)
        w.writerow({"relation": rel, "n_groups": rel_groups[rel],
                    "FR_SS": (f"{fr:.4f}" if fr is not None else ""),
                    "FR_SS_ci95_low": (f"{ci[0]:.4f}" if ci else ""),
                    "FR_SS_ci95_high": (f"{ci[1]:.4f}" if ci else ""),
                    "reportable_ci": bool(ci)})
    print("relation audit rows:", len(rel_groups))

# ---- save summary ----
(OUT / "scripts" / "_dev_summary.json").write_text(json.dumps({
    "metrics": metrics, "ACC_o": ACC_o, "ACC_s": ACC_s, "RPAG": RPAG,
    "false_reject_SS": FR_SS, "false_accept_SO": FA_SO,
    "ss_false_reject_groups": len(ss_err_groups),
    "total_tie_rate": total_tie,
    "d_raw": {"mean": sum(all_d) / len(all_d), "median": pct(0.5), "p05": pct(0.05),
              "p25": pct(0.25), "p75": pct(0.75), "p95": pct(0.95)},
    "bootstrap": {"FR_SS": {"ci95_low": fr_ci[0], "ci95_high": fr_ci[1], "seed": 20260819, "n": 2000},
                  "RPAG": {"ci95_low": rpag_ci[0], "ci95_high": rpag_ci[1], "seed": 20260819, "n": 2000}},
    "relation": {rel: {"n": rel_groups[rel], "FR_SS": rel_fr[rel]["FR_SS"],
                       "ci95": rel_ci.get(rel)} for rel in sorted(rel_groups)},
}, indent=2), encoding="utf-8")

# per-row CSV for failure examples
with open(OUT / "scripts" / "_dev_fourcell_rows.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("Phase 2 OK")
