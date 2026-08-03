#!/usr/bin/env python3
"""D4-M0 Phase 2: T0 reference-adherence behavior qualification on dev 195 groups.

For each dev group × {OO, OS, SO, SS} with T0 template:
  l_A, l_B, d_raw, prediction, cell correctness (teacher-forced, pos=prompt_len-1).
Metrics: accuracy_OO/OS/SO/SS, ACC_o, ACC_s, RPAG, SS_false_rejection_rate,
  SO_false_acceptance_rate, SS_error_group_count, ties, NaN/inf count.
Group-level bootstrap (2000 resamples, seed=20260811) -> 95% CI for RPAG and
  SS_false_rejection_rate.

T0 decision gate: uniquely recovered from D1 original protocol (d1_gate.py):
  c1 readout regression passed (Phase 1)
  c2 ACC_o >= 0.85
  c3 RPAG >= 0.15
  c4 (SS err groups >= 50 AND ss_err_rate >= 0.25) OR (SO err groups >= 50 AND so_err_rate >= 0.25)
  c5 total tie_rate <= 0.02
  c6 no NaN/truncation/parse failure, rows == 780

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


def main():
    tok, model = load_model()
    tok_a = tok.encode(TOK_A, add_special_tokens=False)
    tok_b = tok.encode(TOK_B, add_special_tokens=False)
    accept_id, reject_id = tok_a[0], tok_b[0]
    assert accept_id != reject_id, "accept/reject ids collide"

    pairs = load_dev_pairs()
    print(f"dev groups: {len(pairs)}")

    rows = []
    t0 = time.time()
    for i, p in enumerate(pairs):
        q, cells = four_cell_rows(p, "T0")
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
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pairs)} groups ({time.time()-t0:.0f}s)")
    print(f"scoring done in {time.time()-t0:.0f}s; rows={len(rows)}")

    # ---- write metrics_by_cell csv ----
    with open(R / "t0_metrics_by_cell_dev.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- per-cell accuracy ----
    acc = {}
    for cell in ["OO", "OS", "SO", "SS"]:
        cell_rows = [r for r in rows if r["cell"] == cell]
        acc[cell] = sum(1 for r in cell_rows if r["correct"]) / len(cell_rows)
    ACC_o = (acc["OO"] + acc["OS"]) / 2
    ACC_s = (acc["SO"] + acc["SS"]) / 2
    RPAG = ACC_o - ACC_s
    FR_SS = 1 - acc["SS"]
    FA_SO = 1 - acc["SO"]

    # group-level error sets
    ss_err_groups = set()
    so_err_groups = set()
    for r in rows:
        if r["cell"] == "SS" and not r["correct"]:
            ss_err_groups.add(r["source_group_id"])
        if r["cell"] == "SO" and not r["correct"]:
            so_err_groups.add(r["source_group_id"])
    ss_err_rate = len(ss_err_groups) / len(pairs)
    so_err_rate = len(so_err_groups) / len(pairs)
    ties = sum(1 for r in rows if r["predicted_label"] == "TIE")
    tie_rate = ties / len(rows)
    nan_inf = sum(1 for r in rows if not math.isfinite(r["d_raw"]))

    print(f"\naccuracy: OO={acc['OO']:.4f} OS={acc['OS']:.4f} SO={acc['SO']:.4f} SS={acc['SS']:.4f}")
    print(f"ACC_o={ACC_o:.4f} ACC_s={ACC_s:.4f} RPAG={RPAG:.4f}")
    print(f"FR_SS={FR_SS:.4f} FA_SO={FA_SO:.4f}")
    print(f"SS err groups={len(ss_err_groups)} ({ss_err_rate:.4f})  SO err groups={len(so_err_groups)} ({so_err_rate:.4f})")
    print(f"ties={ties} ({tie_rate:.4f})  nan/inf={nan_inf}")

    # ---- group-level bootstrap (2000 resamples, seed=20260811) ----
    rng = np.random.RandomState(20260811)
    B = 2000
    n = len(pairs)
    # per-group vectors
    grp = []
    for p in pairs:
        gid = p["original_group_id"]
        grp.append({
            "gid": gid,
            "OO": any(r["correct"] for r in rows if r["source_group_id"] == gid and r["cell"] == "OO"),
            "OS": any(r["correct"] for r in rows if r["source_group_id"] == gid and r["cell"] == "OS"),
            "SO": any(r["correct"] for r in rows if r["source_group_id"] == gid and r["cell"] == "SO"),
            "SS": any(r["correct"] for r in rows if r["source_group_id"] == gid and r["cell"] == "SS"),
        })
    rpags = np.empty(B)
    fr_ss = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        gg = [grp[i] for i in idx]
        acc_o = (sum(g["OO"] for g in gg) + sum(g["OS"] for g in gg)) / (2 * n)
        acc_s = (sum(g["SO"] for g in gg) + sum(g["SS"] for g in gg)) / (2 * n)
        rpags[b] = acc_o - acc_s
        fr_ss[b] = 1 - (sum(g["SS"] for g in gg) / n)
    def ci(x):
        lo = np.percentile(x, 2.5)
        hi = np.percentile(x, 97.5)
        return float(lo), float(hi)
    rpag_ci = ci(rpags)
    fr_ci = ci(fr_ss)
    print(f"bootstrap RPAG 95% CI = [{rpag_ci[0]:.4f}, {rpag_ci[1]:.4f}]")
    print(f"bootstrap FR_SS 95% CI = [{fr_ci[0]:.4f}, {fr_ci[1]:.4f}]")

    # ---- bootstrap csv ----
    with open(R / "bootstrap_behavior_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase", "template", "metric", "point", "ci95_low", "ci95_high", "seed", "n_resamples"])
        w.writerow(["phase2", "T0", "RPAG", RPAG, rpag_ci[0], rpag_ci[1], 20260811, B])
        w.writerow(["phase2", "T0", "SS_false_rejection_rate", FR_SS, fr_ci[0], fr_ci[1], 20260811, B])

    # ---- T0 decision gate (inherited from D1) ----
    c1 = True  # Phase 1 readout regression passed (verified in phase1)
    c2 = ACC_o >= 0.85
    c3 = RPAG >= 0.15
    c4_ss = (len(ss_err_groups) >= 50) and (ss_err_rate >= 0.25)
    c4_so = (len(so_err_groups) >= 50) and (so_err_rate >= 0.25)
    c4 = c4_ss or c4_so
    c5 = tie_rate <= 0.02
    c6 = (nan_inf == 0) and (len(rows) == 780)
    conditions = {"c1_readout": c1, "c2_ACC_o_ge_0.85": c2, "c3_RPAG_ge_0.15": c3,
                  "c4_error_volume": c4, "c5_tie_rate": c5, "c6_no_nan": c6}
    print("T0 gate conditions:", json.dumps(conditions, indent=2))

    # ---- save gate summary json (for phase 4/decision) ----
    summary = {
        "phase": "phase2", "template": "T0",
        "acc_by_cell": acc, "ACC_o": ACC_o, "ACC_s": ACC_s, "RPAG": RPAG,
        "false_reject_SS": FR_SS, "false_accept_SO": FA_SO,
        "ss_error_groups": len(ss_err_groups), "so_error_groups": len(so_err_groups),
        "ss_error_rate": ss_err_rate, "so_error_rate": so_err_rate,
        "tie_rate": tie_rate, "nan_inf": nan_inf,
        "bootstrap_RPAG_ci": rpag_ci, "bootstrap_FR_SS_ci": fr_ci,
        "gate_conditions": conditions, "gate_source": "D1 d1_gate.py (unique recovery)",
    }
    (R / "scripts" / "_phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not all(conditions.values()):
        fail("mistral_reference_adherence_behavior_insufficient",
             f"T0 gate failed: {json.dumps(conditions)}")
    print("\nT0 GATE PASSED -> proceed to Phase 3")


if __name__ == "__main__":
    main()
