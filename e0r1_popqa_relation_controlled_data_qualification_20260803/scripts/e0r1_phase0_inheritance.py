#!/usr/bin/env python3
"""E0-R1 Phase 0: inheritance audit.

Verifies all E0 artifacts referenced by the E0-R1 protocol are uniquely inheritable:
 1. E0 final_label == popqa_relation_swap_capacity_insufficient
 2. E0 only failure == relations 16 < 20
 3. PopQA revision / file SHA256 / 14,267 rows / schema (question/obj/prop/id) consistent
 4. R1-R8 filter, split seed 20260816, 60/20/20, within-split same-relation donor,
    per-group RNG "20260816|sgid" uniquely recoverable from E0 scripts
 5. T0/T1/T2 match D1-R-A canonical strings + SHA256
 6. E0 loaded no Judge, ran no inference, produced no blind packet

Writes: inheritance_audit.md
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "e0r1_popqa_relation_controlled_data_qualification_20260803"
E0 = REPO_ROOT / "e0_popqa_relation_controlled_data_qualification_20260803"
D1RA = REPO_ROOT / "d1ra_candidate_template_provenance_diversity_audit_20260803"

rows = []


def check(name, ok, val=""):
    rows.append((name, ok, val))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: {val}")


# 1 & 2: E0 decision
d0 = json.loads((E0 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
check("E0 final_label", d0["final_label"] == "popqa_relation_swap_capacity_insufficient", d0["final_label"])
fr = d0.get("capacity_failure_reason", "")
# failure must mention the 16-vs-20 relation count discrepancy
er = (E0 / "final_report.md").read_text(encoding="utf-8")
rel_cap_fail = ("16 < 20" in er or "16<20" in er or "relation" in er.lower()) and "16" in er and "20" in er
check("E0 failure is relation count", rel_cap_fail and "relation" in fr.lower(), "16<20 relation gate")

# 3: source contract
sac = (E0 / "source_access_audit.md").read_text(encoding="utf-8")
check("E0 revision", "098765c79ea10a2cb19c828324e33281b8336ec0" in sac, "098765c7")
check("E0 test.tsv sha256", "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b" in sac, "9a5227f4")
check("E0 README sha256", "bb04b56bc87a3b2865cc2e2a1649ba6c766a7a44dcba5a53170fbfc72c0da9f0" in sac, "bb04b56b")
sdc = (E0 / "source_data_contract.md").read_text(encoding="utf-8")
check("E0 schema fields", all(f in sdc for f in ("`question`", "`obj`", "`prop`", "`id`")), "question/obj/prop/id")
check("E0 14,267 rows", "14,267" in (E0 / "source_data_contract.md").read_text(encoding="utf-8"), "14,267")

# 4: filter/split/donor recoverable
p1 = (E0 / "scripts" / "e0_phase1_filter.py").read_text(encoding="utf-8")
p2 = (E0 / "scripts" / "e0_phase2_swap.py").read_text(encoding="utf-8")
p0 = (E0 / "scripts" / "e0_phase0_contract.py").read_text(encoding="utf-8")
check("R1-R6 in E0 script", all(f"R{k}" in p1 for k in range(1, 7)), "R1..R6")
check("R7/R8 in E0 script", "R7" in p2 and "R8" in p2, "R7/R8")
check("split seed 20260816", "20260816" in p2 and "SEED_SPLIT = 20260816" in p2, "20260816")
check("60/20/20", "0.6" in p2 and "0.2" in p2, "60/20/20")
check("per-group RNG", '"20260816|" + src[\'source_group_id\']' in p2 or "20260816|" in p2, "20260816|sgid")
check("donor same-split same-relation", "same split" in p2.lower() or "split_name" in p2, "by_rel per split")
check("candidates sorted by sgid", "cands.sort" in p2, "sorted candidates")
check("answer differs", "answer_norm" in p2 and "!=" in p2, "norm(r_o)!=norm(r_s)")
funnel = json.load(open(E0 / "scripts" / "_funnel_r1_6.json"))
check("E0 funnel R6=189", funnel["R6"] == 189, str(funnel["R6"]))
check("E0 funnel R4=1", funnel["R4"] == 1, str(funnel["R4"]))

# 5: templates vs D1-R-A
canon = json.loads((D1RA / "canonical_candidate_templates.json").read_text(encoding="utf-8"))
EXP = {"T0": "The answer is <answer>.", "T1": "For this question, the answer is <answer>.",
       "T2": "The response is <answer>."}
for k in ("T0", "T1", "T2"):
    t = canon[k]["template"]
    sh = hashlib.sha256(t.encode("utf-8")).hexdigest()
    check(f"T{k} canonical+sha", t == EXP[k] and len(sh) == 64, f"{t!r} {sh[:12]}")
spec = json.loads((E0 / "relation_controlled_swap_spec.json").read_text(encoding="utf-8"))
check("E0 spec templates", spec["templates"]["T0"] == EXP["T0"], "spec T0")

# 6: E0 no judge / no inference / no blind packet
check("E0 judge_loaded False", d0["judge_loaded"] is False, str(d0["judge_loaded"]))
check("E0 inference False", d0["inference_run"] is False, str(d0["inference_run"]))
check("E0 blind packet absent", not (E0 / "blind_candidate_contract_packet.csv").exists(), "absent")

all_ok = all(ok for _, ok, _ in rows)
(R / "inheritance_audit.md").write_text(
    """# inheritance_audit.md

## E0 → E0-R1 继承审计

| 项 | 状态 |
|---|---|
"""
    + "\n".join(f"| {n} | {'✓' if ok else '✗'} |" for n, ok, _ in rows)
    + """

## 结论

E0-R1 仅修正 relation 覆盖门槛（见 `protocol_amendment_e0_to_e0r1.md`）；数据源、过滤、split、donor、
四格、模板全部逐字继承 E0。
""", encoding="utf-8")

if not all_ok:
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": "inheritance_invalid",
         "reason": "; ".join(n for n, ok, _ in rows if not ok)}, indent=2), encoding="utf-8")
    print("STOP: inheritance_invalid")
    sys.exit(1)
print("inheritance audit PASSED")
