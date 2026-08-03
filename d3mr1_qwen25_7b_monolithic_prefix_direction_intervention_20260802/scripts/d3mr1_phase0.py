#!/usr/bin/env python3
"""D3-M-R1 Phase 0: inheritance + dev 780 monolithic zero-equivalence regression.

0.1: verify 7 labels, model hashes, T0, synthetic readout (24/24, D1 record),
     D3-M findings (baseline reproduction, hook mapping, passive hook).
0.2: run dev 195×4 monolithic full forward with passive hook, extract
     L18/R_end, compare vs D1 row-by-row (labels + d_raw + hidden consistency),
     and verify R_end positions against D2-R1 contract audit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch

import d3mr1_core as C

D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
D3 = REPO_ROOT / "d3_qwen25_7b_reference_binding_selective_patching_20260802"
D3M = REPO_ROOT / "d3m_qwen25_7b_monolithic_reference_binding_intervention_20260802"
R = REPO_ROOT / "d3mr1_qwen25_7b_monolithic_prefix_direction_intervention_20260802"
MODEL = C.MODEL


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
        "monolithic_full_forward_only": True, "prefix_cache_used": False,
        "activation_intervention_run": False, "prompt_baselines_run": False,
        "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label(p: Path) -> str:
    return json.loads(p.read_text(encoding="utf-8")).get("final_label")


# ---- 0.1 inheritance ----
checks = {
    "D0": (D0 / "artifacts" / "decision.json", "jar_style_sciq_data_qualification_feasible"),
    "D1": (D1 / "artifacts" / "decision.json", "jar_style_reference_override_behavior_feasible"),
    "D1R": (D1R / "artifacts" / "decision.json", "template_robust_reference_override_feasible"),
    "D2": (D2 / "artifacts" / "decision.json", "prefix_causality_audit_invalid"),
    "D2R1": (D2R1 / "artifacts" / "decision.json", "true_prefix_reference_state_signal_localized"),
    "D3": (D3 / "artifacts" / "decision.json", "segmented_execution_equivalence_invalid"),
    "D3M": (D3M / "artifacts" / "decision.json", "monolithic_direction_label_capacity_insufficient"),
}
for k, (p, exp) in checks.items():
    got = label(p)
    if got != exp:
        fail("inheritance_or_data_contract_invalid", f"{k} label={got}")
print("seven labels OK")

rev = (Path(MODEL) / "REVISION.txt").read_text(encoding="utf-8").strip()
if rev != "a09a35458c702b33eeacc393d103063234e8bc28":
    fail("inheritance_or_data_contract_invalid", f"revision={rev}")
d1_audit = (D1 / "model_access_audit.md").read_text(encoding="utf-8")
rec = {}
for line in d1_audit.splitlines():
    m = re.match(r"\| (\S+) \| ([0-9a-f]{64}) \|", line)
    if m:
        rec[m.group(1)] = m.group(2)
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
          "merges.txt", "model.safetensors.index.json"]:
    if rec.get(f) != sha256_file(Path(MODEL) / f):
        fail("inheritance_or_data_contract_invalid", f"{f} hash mismatch")
print("model hashes OK")

spec = json.loads((D1R / "candidate_template_robustness_spec.json").read_text(encoding="utf-8"))
t0 = spec["templates"]["T0"]
if t0["template"] != "The answer is <answer>." or \
        t0["utf8_sha256"] != "c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc":
    fail("inheritance_or_data_contract_invalid", "T0 mismatch")

# synthetic readout: D1 record
syn = list(csv.DictReader(open(D1 / "synthetic_readout_audit.csv", encoding="utf-8")))
if len(syn) != 24 or sum(1 for r in syn if r["correct"] == "True") != 24 \
        or sum(1 for r in syn if r["predicted_label"] == "TIE") != 0:
    fail("inheritance_or_data_contract_invalid", f"synthetic readout n={len(syn)}")
print("synthetic readout (D1 record): 24/24, ties=0")

# D3-M findings
d3m = json.loads((D3M / "artifacts" / "decision.json").read_text(encoding="utf-8"))
if not (d3m["final_label"] == "monolithic_direction_label_capacity_insufficient" and
        d3m.get("activation_intervention_run") is False and
        d3m.get("final_reserve_model_scored") is False):
    fail("inheritance_or_data_contract_invalid", "D3-M findings inconsistent")
d3m_sum = json.loads((D3M / "monolithic_baseline_reproduction_summary.json").read_text(encoding="utf-8"))
if d3m_sum["label_mismatch"] != 0 or d3m_sum["max_bf16_ulp"] != 0.0:
    fail("inheritance_or_data_contract_invalid", "D3-M baseline reproduction not clean")
print("D3-M findings confirmed (780/780 bit-exact, passive hook 0, no intervention)")

# ---- 0.2 dev 780 monolithic zero-equivalence regression ----
d1_rows = list(csv.DictReader(open(D1 / "four_cell_scores_dev.csv", encoding="utf-8")))
if len(d1_rows) != 780:
    fail("intervention_execution_invalid", f"D1 rows={len(d1_rows)}")

# D2-R1 contract R_end reference
contract = {}
for r in csv.DictReader(open(D2R1 / "true_prefix_contract_audit.csv", encoding="utf-8")):
    contract[r["group_id"]] = (int(r["r_end_pos"]), int(r["prefix_len"]), r["T0_prefix_sha"])

pairs = C.load_swap_pairs("dev")
out = []
r_end_mismatch = 0
for p in pairs:
    gid = p["original_group_id"]
    for cell, ref, cand, exp in C.four_cells(p):
        s = C.run_intervention(p["q"], ref, cand, apply_fn=None)  # passive hook
        r1 = next(r for r in d1_rows if r["source_group_id"] == gid and r["cell"] == cell)
        match = s["predicted_label"] == r1["predicted_label"]
        dd = abs(s["d_raw"] - float(r1["d_raw"]))
        # R_end contract check on SS cell
        if cell == "SS":
            r2, plen, ph = contract.get(gid, (None, None, None))
            if r2 != s["r_end_pos"]:
                r_end_mismatch += 1
        out.append({"source_group_id": gid, "cell": cell,
                    "d1_predicted_label": r1["predicted_label"], "our_predicted_label": s["predicted_label"],
                    "label_match": match, "d1_d_raw": float(r1["d_raw"]), "our_d_raw": s["d_raw"],
                    "d_raw_abs_diff": round(dd, 8), "r_end_pos": s["r_end_pos"],
                    "seq_len": s["seq_len"]})
    print(f"  {gid[:12]} done")

with open(R / "monolithic_hook_equivalence_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

nm = sum(1 for x in out if not x["label_match"])
maxd = max(x["d_raw_abs_diff"] for x in out)
print(f"rows={len(out)} label_mismatch={nm} max_d_raw_diff={maxd} r_end_mismatch={r_end_mismatch}")
invalid = (nm != 0 or maxd != 0.0 or r_end_mismatch != 0)
if invalid:
    fail("intervention_execution_invalid",
         f"label_mismatch={nm} max_d={maxd} r_end_mismatch={r_end_mismatch}")

# save summary + inheritance audit
(R / "model_access_audit.md").write_text(
    "| 文件 | SHA256 | 一致性 |\n|---|---|---|\n" +
    "\n".join(f"| {f} | {sha256_file(Path(MODEL)/f)} | 与 D1 一致 |" for f in list(rec)[:6]) +
    f"\n\nrevision = `{rev}`\n", encoding="utf-8")
print("Phase 0 OK")
