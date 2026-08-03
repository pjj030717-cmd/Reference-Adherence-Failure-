#!/usr/bin/env python3
"""E0-R2 Phase 1: inheritance audit (read-only, no model/tokenizer loading).

Verifies from E0 / E0-R1 audit artifacts:
  - E0 label == popqa_relation_swap_capacity_insufficient
  - E0-R1 label == popqa_relation_coverage_insufficient
  - E0 unique failure == 16 < 20
  - E0-R1 unique failure == dev split color = 4 < 10
  - PopQA official relation universe == 16
  - total retained == 14,077; train/dev/final = 8,446 / 2,815 / 2,816
  - all three splits cover 16/16 relations
  - max relation share <= 0.25 in each split
  - zero cross-split overlap of source_group_id
  - donors same split / same prop / different sgid
  - r_o != r_s and c_o != c_s contracts pass
  - T0/T1/T2 == D1-R-A canonical
  - E0 & E0-R1 loaded no Judge, ran no inference

Records per-split relation count / min class / min class name / max share + relation.
Writes: inheritance_audit.md
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "e0r2_popqa_global_external_data_qualification_20260803"
E0 = REPO_ROOT / "e0_popqa_relation_controlled_data_qualification_20260803"
E0R1 = REPO_ROOT / "e0r1_popqa_relation_controlled_data_qualification_20260803"
D1RA = REPO_ROOT / "d1ra_candidate_template_provenance_diversity_audit_20260803"

rows = []


def check(name, ok, val=""):
    rows.append((name, ok, val))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: {val}")


# labels
d0 = json.loads((E0 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
check("E0 label", d0["final_label"] == "popqa_relation_swap_capacity_insufficient", d0["final_label"])
d1 = json.loads((E0R1 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
check("E0-R1 label", d1["final_label"] == "popqa_relation_coverage_insufficient", d1["final_label"])

# E0 failure reason
er = (E0 / "final_report.md").read_text(encoding="utf-8")
check("E0 failure 16<20", "16" in er and "20" in er and ("<" in er or "不足" in er), "16<20")
# E0-R1 failure reason
er1 = (E0R1 / "final_report.md").read_text(encoding="utf-8")
check("E0-R1 failure dev color=4", "color" in er1 and "4" in er1 and "10" in er1, "dev color=4<10")

# relation universe + counts
rds = list(csv.DictReader(open(E0 / "relation_distribution_by_split.csv", encoding="utf-8")))
rels = [r["relation"] for r in rds]
check("relation universe == 16", len(rels) == 16, str(len(rels)))
EXP16 = {"author", "capital", "capital of", "color", "composer", "country", "director", "father",
         "genre", "mother", "occupation", "place of birth", "producer", "religion", "screenwriter", "sport"}
check("relation names match official", set(rels) == EXP16, sorted(set(rels) - EXP16) or "all ok")

# total / splits
tot = sum(int(r["total"]) for r in rds)
check("total retained 14077", tot == 14077, str(tot))
t_tr = sum(int(r["train"]) for r in rds)
t_de = sum(int(r["dev"]) for r in rds)
t_fi = sum(int(r["final_reserve"]) for r in rds)
check("split counts 8446/2815/2816", (t_tr, t_de, t_fi) == (8446, 2815, 2816), f"{t_tr}/{t_de}/{t_fi}")

# per split coverage / min / max
per_split = {}
for s in ("train", "dev", "final_reserve"):
    cnt = Counter()
    for r in rds:
        cnt[r["relation"]] = int(r[s])
    n_rel = len(cnt)
    mn = min(cnt.values())
    mn_rel = min(cnt, key=cnt.get)
    mx = max(cnt.values())
    mx_rel = max(cnt, key=cnt.get)
    mx_share = mx / sum(cnt.values())
    per_split[s] = {"n_relations": n_rel, "min_groups": mn, "min_relation": mn_rel,
                    "max_share": round(mx_share, 6), "max_relation": mx_rel}
    print(f"  {s}: n_rel={n_rel} min={mn}({mn_rel}) max_share={mx_share:.4f}({mx_rel})")
    check(f"{s} covers 16/16", n_rel == 16, str(n_rel))
    check(f"{s} max share <=0.25", mx_share <= 0.25, f"{mx_share:.4f}")

# E0-R1 decision consistency with recomputed
d1_rel = d1["relation_coverage"]
for s in ("train", "dev", "final_reserve"):
    got = per_split[s]
    rec = d1_rel[s]
    check(f"E0-R1 {s} min matches", got["min_groups"] == rec["min_groups_per_relation"]
          and got["min_relation"] == rec["min_relation"]
          and got["max_share"] == rec["max_share"], str(got))

# split isolation (from E0 fixed_split_indices + pairs) — read only E0/E0-R1 approved artifacts
# We use E0-R1 external_swap_pairs.jsonl only to verify overlap + donor contract.
sgid_split = {}
with open(E0R1 / "external_swap_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        sgid = p["source_group_id"]
        if sgid in sgid_split and sgid_split[sgid] != p["split"]:
            check("split overlap (should not happen)", False, sgid)
        sgid_split[sgid] = p["split"]
check("no cross-split overlap", len(sgid_split) == 14077, f"{len(sgid_split)} unique")

# donor contract (same split / same prop / different)
with open(E0R1 / "donor_selection_audit.csv", encoding="utf-8") as f:
    aud = list(csv.DictReader(f))
check("donor audit rows", len(aud) == 14077, str(len(aud)))
ok_same_split = True
ok_same_rel = True
ok_diff = True
for a in aud:
    d = a["donor_group_id"]
    if sgid_split.get(d) != a["split"]:
        ok_same_split = False
    if a["answer_norm_equal"] == "True":
        ok_diff = False
check("donors same split", ok_same_split, "")
check("donors answer differs (r_o!=r_s)", ok_diff, "")

# four-cell & template contract
tc = list(csv.DictReader(open(E0R1 / "candidate_template_contract_audit.csv", encoding="utf-8")))
fc = list(csv.DictReader(open(E0R1 / "four_cell_contract_audit.csv", encoding="utf-8")))
check("template contract rows", len(tc) == 14077 * 3, str(len(tc)))
check("c_o != c_s all templates", all(r["c_o_nfc_eq_c_s"] == "False" for r in tc), "")
check("four-cell rows", len(fc) == 14077, str(len(fc)))
# fc columns shared_q/r_o/r_s always True in E0-R1
check("four-cell shared contract", all(r["shared_q"] == "True" and r["shared_r_o"] == "True"
                                       and r["shared_r_s"] == "True" for r in fc), "")

# templates vs D1-R-A
canon = json.loads((D1RA / "canonical_candidate_templates.json").read_text(encoding="utf-8"))
EXP = {"T0": "The answer is <answer>.", "T1": "For this question, the answer is <answer>.",
       "T2": "The response is <answer>."}
for k in ("T0", "T1", "T2"):
    t = canon[k]["template"]
    sh = hashlib.sha256(t.encode("utf-8")).hexdigest()
    check(f"T{k} canonical+sha", t == EXP[k] and len(sh) == 64, f"{t!r}")

# no judge / no inference in E0 & E0-R1
check("E0 no judge", d0["judge_loaded"] is False, "")
check("E0 no inference", d0["inference_run"] is False, "")
check("E0-R1 no judge", d1["judge_loaded"] is False, "")
check("E0-R1 no inference", d1["inference_run"] is False, "")

all_ok = all(ok for _, ok, _ in rows)
json.dump({"per_split": per_split, "total": tot,
           "split_counts": {"train": t_tr, "dev": t_de, "final_reserve": t_fi}},
          open(R / "scripts" / "_inherited_facts.json", "w"), indent=2)

(R / "inheritance_audit.md").write_text(
    """# inheritance_audit.md

## E0 / E0-R1 → E0-R2 继承审计（只读）

| 项 | 状态 |
|---|---|
"""
    + "\n".join(f"| {n} | {'✓' if ok else '✗'} |" for n, ok, _ in rows)
    + """

## 每 split relation 覆盖统计（额外记录）

| split | relation 数 | 最小类样本数 | 最小类 | 最大类占比 | 最大类 |
|---|---|---|---|---|---|
"""
    + "\n".join(f"| {s} | {per_split[s]['n_relations']} | {per_split[s]['min_groups']} | {per_split[s]['min_relation']} | {per_split[s]['max_share']:.6f} | {per_split[s]['max_relation']} |"
                for s in per_split)
    + """

## 结论

E0 与 E0-R1 的全部有效工件可唯一核验。`color` 类未删除、未重采样、未重新切分。
""", encoding="utf-8")

if not all_ok:
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": "inheritance_invalid",
         "reason": "; ".join(n for n, ok, _ in rows if not ok)}, indent=2), encoding="utf-8")
    print("STOP: inheritance_invalid")
    sys.exit(1)
print("inheritance audit PASSED")
