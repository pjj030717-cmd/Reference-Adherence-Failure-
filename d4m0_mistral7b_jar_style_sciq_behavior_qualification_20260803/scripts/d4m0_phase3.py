#!/usr/bin/env python3
"""D4-M0 Phase 3: T1/T2 template robustness on the same 195 dev groups.

For T1 and T2, compute the same four-cell metrics as T0, plus SS error retention
vs T0 (T0_SS_error_set ∩ T1_SS_error_set / |T0_SS_error_set|).
Group-level bootstrap (2000 resamples, seed=20260811) for FR_SS 95% CI and
retention 95% CI.

Gate inherited verbatim from D1-R (d1r_gate.py, uniquely recovered):
  for each of T1, T2:
    ACC_o >= 0.95
    false_reject_SS >= 0.50
    FR bootstrap CI lower >= 0.40
    RPAG >= 0.20
    SS_error_retention >= 0.60
    total tie_rate <= 0.02
    no NaN / truncation / parse failure

Only D0 dev (via D1 dev-only file) is streamed; no train / final-reserve text.
No hidden states are extracted or saved.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from d4m0_core import R, load_model, load_dev_pairs, build_messages, score_prompt, classify, four_cell_rows

TOK_A = " A"
TOK_B = " B"
BOOT_SEED = 20260811
BOOT_N = 2000

def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(
        json.dumps({"final_label": label, "reason": why,
                    "final_reserve_read": False, "hidden_states_read": False,
                    "probe_trained": False, "activation_intervention_run": False,
                    "prompt_baselines_run": False, "train_text_read": False}, indent=2),
        encoding="utf-8")
    sys.exit(1)


def compute_template(tok, model, pairs, accept_id, reject_id, template: str):
    rows = []
    for p in pairs:
        q, cells = four_cell_rows(p, template)
        for cell, ref, cand, exp in cells:
            msgs = build_messages(q, ref, cand)
            l_A, l_B, d_raw, p_acc, plen, gid, gtok = score_prompt(tok, model, msgs, accept_id, reject_id)
            pred = classify(d_raw)
            rows.append({
                "source_group_id": p["original_group_id"],
                "cell": cell, "question": q, "reference": ref, "candidate": cand,
                "expected_label": exp, "l_A": l_A, "l_B": l_B, "d_raw": d_raw,
                "p_accept_raw": p_acc, "prompt_len": plen,
                "predicted_label": pred, "correct": pred == exp,
            })
    return rows


def summarize(rows, pairs):
    acc = {}
    for cell in ["OO", "OS", "SO", "SS"]:
        cell_rows = [r for r in rows if r["cell"] == cell]
        acc[cell] = sum(1 for r in cell_rows if r["correct"]) / len(cell_rows)
    ACC_o = (acc["OO"] + acc["OS"]) / 2
    ACC_s = (acc["SO"] + acc["SS"]) / 2
    RPAG = ACC_o - ACC_s
    FR_SS = 1 - acc["SS"]
    FA_SO = 1 - acc["SO"]
    ss_err = set()
    so_err = set()
    for r in rows:
        if r["cell"] == "SS" and not r["correct"]:
            ss_err.add(r["source_group_id"])
        if r["cell"] == "SO" and not r["correct"]:
            so_err.add(r["source_group_id"])
    ties = sum(1 for r in rows if r["predicted_label"] == "TIE")
    nan_inf = sum(1 for r in rows if not math.isfinite(r["d_raw"]))
    return {"acc_by_cell": acc, "ACC_o": ACC_o, "ACC_s": ACC_s, "RPAG": RPAG,
            "false_reject_SS": FR_SS, "false_accept_SO": FA_SO,
            "ss_error_groups": len(ss_err), "so_error_groups": len(so_err),
            "ss_error_set": ss_err, "tie_rate": ties / len(rows), "nan_inf": nan_inf}


def bootstrap(rows, pairs, seed=BOOT_SEED, n=BOOT_N):
    """Group-level bootstrap of FR_SS and RPAG. Returns (fr_ci, rpag_ci)."""
    rng = np.random.RandomState(seed)
    grp = []
    for p in pairs:
        gid = p["original_group_id"]
        grp.append({"gid": gid,
                    "OO": any(r["correct"] for r in rows if r["source_group_id"] == gid and r["cell"] == "OO"),
                    "OS": any(r["correct"] for r in rows if r["source_group_id"] == gid and r["cell"] == "OS"),
                    "SO": any(r["correct"] for r in rows if r["source_group_id"] == gid and r["cell"] == "SO"),
                    "SS": any(r["correct"] for r in rows if r["source_group_id"] == gid and r["cell"] == "SS")})
    N = len(pairs)
    rpags = np.empty(n)
    frs = np.empty(n)
    for b in range(n):
        idx = rng.randint(0, N, size=N)
        gg = [grp[i] for i in idx]
        acc_o = (sum(g["OO"] for g in gg) + sum(g["OS"] for g in gg)) / (2 * N)
        acc_s = (sum(g["SO"] for g in gg) + sum(g["SS"] for g in gg)) / (2 * N)
        rpags[b] = acc_o - acc_s
        frs[b] = 1 - (sum(g["SS"] for g in gg) / N)
    return (float(np.percentile(frs, 2.5)), float(np.percentile(frs, 97.5))), \
           (float(np.percentile(rpags, 2.5)), float(np.percentile(rpags, 97.5)))


def main():
    tok, model = load_model()
    tok_a = tok.encode(TOK_A, add_special_tokens=False)
    tok_b = tok.encode(TOK_B, add_special_tokens=False)
    accept_id, reject_id = tok_a[0], tok_b[0]

    pairs = load_dev_pairs()
    print(f"dev groups: {len(pairs)}")

    # T0 SS error set from Phase 2 summary
    ph2 = json.loads((R / "scripts" / "_phase2_summary.json").read_text(encoding="utf-8"))
    t0_rows = list(csv.DictReader(open(R / "t0_metrics_by_cell_dev.csv", encoding="utf-8")))
    t0_ss_err = set()
    for r in t0_rows:
        if r["cell"] == "SS" and r["correct"] == "False":
            t0_ss_err.add(r["source_group_id"])
    n_t0_ss = len(t0_ss_err)
    print(f"T0 SS error set size: {n_t0_ss}")

    results = {}
    retention = {}
    for template in ["T1", "T2"]:
        t0 = time.time()
        rows = compute_template(tok, model, pairs, accept_id, reject_id, template)
        print(f"  {template}: {len(rows)} rows in {time.time()-t0:.0f}s")

        s = summarize(rows, pairs)
        fr_ci, rpag_ci = bootstrap(rows, pairs)
        ret = len(t0_ss_err & s["ss_error_set"]) / n_t0_ss if n_t0_ss else float("nan")
        results[template] = {"summary": s, "fr_ci": fr_ci, "rpag_ci": rpag_ci, "retention": ret}
        retention[template] = ret
        print(f"  {template}: ACC_o={s['ACC_o']:.4f} RPAG={s['RPAG']:.4f} FR_SS={s['false_reject_SS']:.4f} "
              f"FA_SO={s['false_accept_SO']:.4f} SS_err={s['ss_error_groups']} SO_err={s['so_error_groups']} "
              f"ties={s['tie_rate']:.4f} nan={s['nan_inf']} ret={ret:.4f}")
        print(f"  {template}: FR_SS 95% CI=[{fr_ci[0]:.4f}, {fr_ci[1]:.4f}] RPAG CI=[{rpag_ci[0]:.4f}, {rpag_ci[1]:.4f}]")

        # save per-template rows
        tmp = R / f"t1_t2_metrics_by_cell_dev.csv"
        header = not tmp.exists() or tmp.stat().st_size == 0
        with open(tmp, "a" if not header else "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["template"] + list(rows[0].keys()))
            if header:
                w.writeheader()
            for r in rows:
                w.writerow({"template": template, **r})

    # ---- retention audit csv ----
    t0_rows_by_gid = {r["source_group_id"]: r for r in t0_rows if r["cell"] == "SS"}
    t1_rows = list(csv.DictReader(open(R / "t1_t2_metrics_by_cell_dev.csv", encoding="utf-8")))
    ret_rows = []
    for gid in sorted(t0_ss_err):
        row = {"source_group_id": gid, "question": t0_rows_by_gid[gid]["question"],
               "T0_SS_d_raw": t0_rows_by_gid[gid]["d_raw"],
               "T0_SS_correct": t0_rows_by_gid[gid]["correct"]}
        for template in ["T1", "T2"]:
            m = [r for r in t1_rows if r["template"] == template and r["cell"] == "SS"
                 and r["source_group_id"] == gid]
            row[f"{template}_SS_d_raw"] = m[0]["d_raw"]
            row[f"{template}_SS_correct"] = m[0]["correct"]
        ret_rows.append(row)
    with open(R / "template_error_retention_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ret_rows[0].keys()))
        w.writeheader()
        w.writerows(ret_rows)

    # ---- combined bootstrap csv (T0 from phase2 + T1/T2) ----
    boot_rows = []
    boot_rows.append(["phase2", "T0", "RPAG", ph2["RPAG"], ph2["bootstrap_RPAG_ci"][0], ph2["bootstrap_RPAG_ci"][1], BOOT_SEED, BOOT_N])
    boot_rows.append(["phase2", "T0", "SS_false_rejection_rate", ph2["false_reject_SS"], ph2["bootstrap_FR_SS_ci"][0], ph2["bootstrap_FR_SS_ci"][1], BOOT_SEED, BOOT_N])
    for template in ["T1", "T2"]:
        r = results[template]
        boot_rows.append(["phase3", template, "RPAG", r["summary"]["RPAG"], r["rpag_ci"][0], r["rpag_ci"][1], BOOT_SEED, BOOT_N])
        boot_rows.append(["phase3", template, "SS_false_rejection_rate", r["summary"]["false_reject_SS"], r["fr_ci"][0], r["fr_ci"][1], BOOT_SEED, BOOT_N])
        boot_rows.append(["phase3", template, "SS_error_retention", r["retention"], r["retention"], r["retention"], BOOT_SEED, BOOT_N])
    with open(R / "bootstrap_behavior_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase", "template", "metric", "point", "ci95_low", "ci95_high", "seed", "n_resamples"])
        w.writerows(boot_rows)

    # ---- gate (inherited from D1-R) ----
    gate = {}
    for template in ["T1", "T2"]:
        r = results[template]
        s = r["summary"]
        gate[template] = {
            "ACC_o_ge_0.95": s["ACC_o"] >= 0.95,
            "FR_SS_ge_0.50": s["false_reject_SS"] >= 0.50,
            "FR_CI_low_ge_0.40": r["fr_ci"][0] >= 0.40,
            "RPAG_ge_0.20": s["RPAG"] >= 0.20,
            "retention_ge_0.60": r["retention"] >= 0.60,
            "tie_rate_le_0.02": s["tie_rate"] <= 0.02,
            "no_nan": s["nan_inf"] == 0,
        }
        print(template, gate[template])

    all_ok = all(all(v for v in gate[t].values()) for t in ["T1", "T2"])
    summary_out = {
        "phase": "phase3",
        "t0_ss_error_groups": n_t0_ss,
        "retention": retention,
        "template_summaries": {t: {k: results[t]["summary"][k] for k in
                                   ["acc_by_cell", "ACC_o", "ACC_s", "RPAG", "false_reject_SS",
                                    "false_accept_SO", "ss_error_groups", "so_error_groups",
                                    "tie_rate", "nan_inf"]} for t in ["T1", "T2"]},
        "bootstrap": {t: {"FR_SS_ci": results[t]["fr_ci"], "RPAG_ci": results[t]["rpag_ci"],
                          "retention": results[t]["retention"]} for t in ["T1", "T2"]},
        "gate_conditions": gate,
        "gate_source": "D1-R d1r_gate.py (unique recovery)",
    }
    (R / "scripts" / "_phase3_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")

    if not all_ok:
        fail("mistral_template_robustness_insufficient", json.dumps(gate))
    print("\nT1/T2 GATE PASSED -> mistral_reference_adherence_behavior_feasible")


if __name__ == "__main__":
    main()
