#!/usr/bin/env python3
"""D1-P Phase 2: 195 dev groups x 4 cells x (baseline x template) behavior eval.

baseline ∈ {B_base, B_direct, B_CoT_gen}
template ∈ {T0, T1, T2}

For B_base / B_direct: teacher-forced A/B readout (d_raw = l_A - l_B).
For B_CoT_gen: greedy generation, parse 'Final verdict: A/B' (last non-empty line, skip chat control tokens).

Outputs:
  - baseline_group_level_verdicts.csv  (per group, per baseline x template: SS verdict & d_raw)
  - baseline_metrics_by_template_cell.csv (per baseline x template x cell aggregate)
  - bbase_reproduction_audit.csv (B_base T0 row-level vs D1 / D1-R)
  - cot_parse_audit.csv (CoT per-group parse summary)
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

R = REPO_ROOT / "d1p_qwen25_7b_prompt_baselines_dev_20260803"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]
spec = json.loads((R / "baseline_prompt_spec.json").read_text(encoding="utf-8"))
USER_TMPL = spec["user_template"]
SYSTEMS = {"B_base": spec["system_base"], "B_direct": spec["B_direct"]["system"], "B_CoT_gen": spec["B_CoT_gen"]["system"]}
TEMPLATES = spec["templates"]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()


def run_tf(system, q, ref, cand):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": USER_TMPL.format(question=q, reference=ref, candidate=cand)}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")["input_ids"].to("cuda")
    plen = ids.shape[1]
    with torch.inference_mode():
        logits = model(ids).logits
    pos = plen - 1
    lA = logits[0, pos, ACCEPT_ID].item()
    lB = logits[0, pos, REJECT_ID].item()
    return lA, lB, lA - lB


def run_cot(system, q, ref, cand):
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": USER_TMPL.format(question=q, reference=ref, candidate=cand)}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"].to("cuda")
    attn = torch.ones_like(ids)
    with torch.inference_mode():
        out = model.generate(ids, attention_mask=attn, do_sample=False, max_new_tokens=128,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    gen = out[0, ids.shape[1]:]
    text = tok.decode(gen, skip_special_tokens=True)
    text = unicodedata.normalize("NFKC", text)
    nonempty = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if nonempty:
        last = nonempty[-1]
        if last == "Final verdict: A":
            return "A", text
        if last == "Final verdict: B":
            return "B", text
    return "UNPARSEABLE", text


dev_pairs = []
for line in open(D1 / "scripts" / "_dev_pairs.jsonl", encoding="utf-8"):
    d = json.loads(line)
    assert d["split"] == "dev"
    dev_pairs.append(d)
print("dev pairs:", len(dev_pairs))
assert len(dev_pairs) == 195

d1_rows = {}
with open(D1 / "scripts" / "_dev_fourcell_rows.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d1_rows[(r["source_group_id"], r["cell"])] = r
d1r_t0 = {}
with open(D1R / "t0_reproduction_audit.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        d1r_t0[(r["source_group_id"], r["cell"])] = r
print("D1 four-cell:", len(d1_rows), "D1-R T0:", len(d1r_t0))

CELLS = [("OO", "A"), ("OS", "B"), ("SO", "B"), ("SS", "A")]
BASELINES = ["B_base", "B_direct", "B_CoT_gen"]

agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
tie_cnt = defaultdict(int)
unp_cnt = defaultdict(int)
repro_rows = []
cot_parse_rows = []

# group-level storage: gid -> baseline -> template -> cell -> {"verdict","d_raw","correct"}
group_fourcell = defaultdict(lambda: defaultdict(dict))

for base in BASELINES:
    system = SYSTEMS[base]
    for tname in ("T0", "T1", "T2"):
        tpl = TEMPLATES[tname]
        for p in dev_pairs:
            gid = p["original_group_id"]
            for cell, exp in CELLS:
                if cell == "OO":
                    ref, ans = p["r_o"], p["c_o"]
                elif cell == "OS":
                    ref, ans = p["r_o"], p["c_s"]
                elif cell == "SO":
                    ref, ans = p["r_s"], p["c_o"]
                else:
                    ref, ans = p["r_s"], p["c_s"]
                cand = ans if tname == "T0" else tpl.replace("<answer>", ans)
                if base == "B_CoT_gen":
                    verdict, text = run_cot(system, p["q"], ref, cand)
                    if verdict == "UNPARSEABLE":
                        unp_cnt[(base, tname)] += 1
                        ok = False
                    else:
                        ok = (verdict == exp)
                    agg[(base, tname)][cell][0] += 1
                    agg[(base, tname)][cell][1] += 1 if ok else 0
                    group_fourcell[gid][f"{base}|{tname}"][cell] = {"verdict": verdict, "d_raw": None, "correct": ok}
                    if verdict == "UNPARSEABLE":
                        cot_parse_rows.append({"source_group_id": gid, "cell": cell, "template": tname,
                                               "parseable": 0, "verdict": "UNPARSEABLE",
                                               "last_line": (text.splitlines()[-1].strip() if text.splitlines() else "")})
                else:
                    lA, lB, d = run_tf(system, p["q"], ref, cand)
                    pred = "A" if d > 0 else ("B" if d < 0 else "TIE")
                    if pred == "TIE":
                        tie_cnt[(base, tname)] += 1
                    ok = (pred == exp)
                    agg[(base, tname)][cell][0] += 1
                    agg[(base, tname)][cell][1] += 1 if ok else 0
                    group_fourcell[gid][f"{base}|{tname}"][cell] = {"verdict": pred, "d_raw": d, "correct": ok}
                    if base == "B_base" and tname == "T0":
                        r1 = d1_rows.get((gid, cell))
                        r2 = d1r_t0.get((gid, cell))
                        d1_pred = r1["predicted_label"] if r1 else None
                        d1r_pred = r2["predicted_label"] if r2 else None
                        repro_rows.append({"source_group_id": gid, "cell": cell, "this_pred": pred,
                                           "d1_pred": d1_pred, "d1_match": (pred == d1_pred) if d1_pred else None,
                                           "d1r_pred": d1r_pred, "d1r_match": (pred == d1r_pred) if d1r_pred else None})
        print(f"  done {base} {tname}")

# ------------------------------------------------------------------
# write per-group four-cell verdicts (all cells)
# ------------------------------------------------------------------
with open(R / "baseline_group_level_verdicts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    hdr = ["source_group_id"]
    for base in BASELINES:
        for tname in ("T0", "T1", "T2"):
            for cell in ("OO", "OS", "SO", "SS"):
                hdr.append(f"{base}_{tname}_{cell}_verdict")
                if base != "B_CoT_gen":
                    hdr.append(f"{base}_{tname}_{cell}_d_raw")
    w.writerow(hdr)
    for p in dev_pairs:
        gid = p["original_group_id"]
        row = [gid]
        for base in BASELINES:
            for tname in ("T0", "T1", "T2"):
                for cell in ("OO", "OS", "SO", "SS"):
                    g = group_fourcell[gid][f"{base}|{tname}"].get(cell, {})
                    row.append(g.get("verdict"))
                    if base != "B_CoT_gen":
                        row.append(g.get("d_raw"))
        w.writerow(row)

# ------------------------------------------------------------------
# write metrics by template cell
# ------------------------------------------------------------------
with open(R / "baseline_metrics_by_template_cell.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["baseline", "template", "cell", "n", "n_correct", "accuracy"])
    for (base, tname), cells in sorted(agg.items()):
        for cell, (n, c) in sorted(cells.items()):
            w.writerow([base, tname, cell, n, c, round(c / n, 6) if n else ""])

# ------------------------------------------------------------------
# write B_base reproduction audit
# ------------------------------------------------------------------
with open(R / "bbase_reproduction_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(repro_rows[0].keys()))
    w.writeheader()
    w.writerows(repro_rows)

# ------------------------------------------------------------------
# write cot parse audit
# ------------------------------------------------------------------
with open(R / "cot_parse_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "cell", "template", "parseable", "verdict", "last_line"])
    w.writeheader()
    w.writerows(cot_parse_rows)

# ------------------------------------------------------------------
# reproduction check: B_base T0 must match D1 (780 rows) and D1-R (780 rows)
# ------------------------------------------------------------------
matches_d1 = sum(1 for r in repro_rows if r["d1_match"] is True)
matches_d1r = sum(1 for r in repro_rows if r["d1r_match"] is True)
total = len(repro_rows)
print(f"B_base T0 reproduction: {matches_d1}/{total} vs D1, {matches_d1r}/{total} vs D1-R")
if matches_d1 != total or matches_d1r != total:
    sys.exit("STOP: B_base T0 does not reproduce D1/D1-R")

# ------------------------------------------------------------------
# group bootstrap (seed 20260815, 2000 iters) on SS FR / RPAG / ACC_o / ACC_s
# ------------------------------------------------------------------
rng = np.random.default_rng(20260815)
N = len(dev_pairs)
gids = [p["original_group_id"] for p in dev_pairs]
B_ITERS = 2000

boot_rows = []
for base in BASELINES:
    for tname in ("T0", "T1", "T2"):
        key = f"{base}|{tname}"
        def metrics_from(inds):
            fr_ss, acc_o, acc_s = [], [], []
            for gi in inds:
                c = group_fourcell[gids[gi]][key]
                oo = c.get("OO", {}).get("correct", False)
                ss = c.get("SS", {}).get("correct", False)
                acc_o.append(1 if oo else 0)
                acc_s.append(1 if ss else 0)
                fr_ss.append(0 if ss else 1)
            m_o, m_s = sum(acc_o) / len(inds), sum(acc_s) / len(inds)
            return sum(fr_ss) / len(inds), m_o - m_s, m_o, m_s
        obs = metrics_from(range(N))
        dist = np.array([metrics_from(rng.integers(0, N, N)) for _ in range(B_ITERS)])
        row = {"baseline": base, "template": tname}
        for name, i in (("SS_FR", 0), ("RPAG", 1), ("ACC_o", 2), ("ACC_s", 3)):
            row[f"{name}_obs"] = round(obs[i], 6)
            row[f"{name}_ci_low"] = round(float(np.percentile(dist[:, i], 2.5)), 6)
            row[f"{name}_ci_high"] = round(float(np.percentile(dist[:, i], 97.5)), 6)
        boot_rows.append(row)

with open(R / "bootstrap_baseline_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
    w.writeheader()
    w.writerows(boot_rows)
for r in boot_rows:
    print(f"boot {r['baseline']} {r['template']}: FR_SS={r['SS_FR_obs']} CI=[{r['SS_FR_ci_low']},{r['SS_FR_ci_high']}] RPAG={r['RPAG_obs']} CI=[{r['RPAG_ci_low']},{r['RPAG_ci_high']}]")

# metrics summary per baseline x template
print("\n=== aggregate metrics ===")
for base in BASELINES:
    for tname in ("T0", "T1", "T2"):
        cells = agg[(base, tname)]
        if not cells:
            continue
        n_o = cells["OO"][0]
        acc_o = cells["OO"][1] / n_o if n_o else float("nan")
        n_s = cells["SS"][0]
        acc_s = cells["SS"][1] / n_s if n_s else float("nan")
        rpag = acc_o - acc_s
        fr_ss = 1 - acc_s  # SS false-rejection rate (SS expects Accept)
        so_fa = 1 - (cells["SO"][1] / cells["SO"][0]) if cells["SO"][0] else float("nan")
        print(f"{base} {tname}: ACC_o={acc_o:.4f} ACC_s={acc_s:.4f} RPAG={rpag:.4f} FR_SS={fr_ss:.4f} SO_fa={so_fa:.4f}")

# stash for deliverables stage
json.dump({"repro_total": total, "matches_d1": matches_d1, "matches_d1r": matches_d1r,
           "agg": {f"{b}|{t}": {c: agg[(b, t)][c] for c in ("OO", "OS", "SO", "SS")} for b in BASELINES for t in ("T0", "T1", "T2")},
           "tie_cnt": {f"{b}|{t}": tie_cnt[(b, t)] for b in BASELINES for t in ("T0", "T1", "T2")},
           "unp_cnt": {f"{b}|{t}": unp_cnt[(b, t)] for b in BASELINES for t in ("T0", "T1", "T2")},
           "boot": boot_rows},
          open(R / "scripts" / "_phase2_agg.json", "w"), indent=2)
print("Phase 2a OK")
