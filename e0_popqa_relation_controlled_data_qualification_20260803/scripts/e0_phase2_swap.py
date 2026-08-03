#!/usr/bin/env python3
"""E0 Phase 2: split-first then relation-controlled donor selection.

2.1 split: sort by source_group_id, shuffle with random.Random(20260816),
           train/dev/final-reserve = 60/20/20.
2.2 donor: within same split, same relation, different source_group_id,
           answer_norm != source answer_norm (R8). Candidates sorted by sgid;
           per-group RNG random.Random("20260816|"+sgid) picks one.
R7: relation with <2 legal records within a split removes those records.
R9: after rendering T0, c_o != c_s (NFKC).

Writes: fixed_split_indices.json, _after_r7.jsonl, donor_assignment_audit.csv,
        external_swap_pairs.jsonl (rows have q, r_o, r_s), funnel extension.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "e0_popqa_relation_controlled_data_qualification_20260803"
T0 = "The answer is <answer>."
SEED_SPLIT = 20260816


def fail(why: str):
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": "popqa_relation_swap_capacity_insufficient", "reason": why}, indent=2), encoding="utf-8")
    print("STOP:", why)
    sys.exit(1)


records = [json.loads(line) for line in open(R / "scripts" / "_after_r6.jsonl", encoding="utf-8")]
print("after R1-R6:", len(records))

# 2.1 split
records_sorted = sorted(records, key=lambda r: r["source_group_id"])
rng = random.Random(SEED_SPLIT)
rng.shuffle(records_sorted)
n = len(records_sorted)
n_train = int(n * 0.6)
n_dev = int(n * 0.2)
n_final = n - n_train - n_dev
train = records_sorted[:n_train]
dev = records_sorted[n_train:n_train + n_dev]
final_reserve = records_sorted[n_train + n_dev:]
print(f"split: train={len(train)} dev={len(dev)} final_reserve={len(final_reserve)}")

# write fixed_split_indices.json: index (position in sorted order) -> split
sorted_ids = [r["source_group_id"] for r in records_sorted]
split_of_idx = {}
for i, r in enumerate(records_sorted):
    if i < n_train:
        split_of_idx[i] = "train"
    elif i < n_train + n_dev:
        split_of_idx[i] = "dev"
    else:
        split_of_idx[i] = "final_reserve"
json.dump({"seed": SEED_SPLIT, "shuffle_applied": True,
           "counts": {"train": len(train), "dev": len(dev), "final_reserve": len(final_reserve)},
           "n_total": n,
           "index_to_split": {str(k): v for k, v in split_of_idx.items()},
           "sorted_source_group_ids": sorted_ids},
          open(R / "fixed_split_indices.json", "w"), indent=2)


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", (s or "")).split())


# per-split relation groups over R1-R6 survivors
r7_excluded = 0
r8_excluded = 0
pairs = []
audit_rows = []

for split_name, split_rows in (("train", train), ("dev", dev), ("final_reserve", final_reserve)):
    by_rel = defaultdict(list)
    for rec in split_rows:
        by_rel[rec["relation_nfkc"]].append(rec)
    # R7: relations with <2 legal records in this split -> exclude those records
    for rel, recs in by_rel.items():
        if len(recs) < 2:
            r7_excluded += len(recs)
            continue
        for src in sorted(recs, key=lambda r: r["source_group_id"]):
            # candidates: same split, same relation, different sgid, answer_norm differs
            cands = [c for c in recs if c["source_group_id"] != src["source_group_id"]
                     and c["answer_norm"] != src["answer_norm"]]
            cands.sort(key=lambda c: c["source_group_id"])
            if not cands:
                r8_excluded += 1
                continue
            grng = random.Random(f"{SEED_SPLIT}|{src['source_group_id']}")
            donor = grng.choice(cands)
            r_s = donor["canonical_answer_nfkc"]
            c_o = T0.replace("<answer>", src["canonical_answer_nfkc"])
            c_s = T0.replace("<answer>", r_s)
            if unicodedata.normalize("NFKC", c_o) == unicodedata.normalize("NFKC", c_s):
                r9_excluded = 1  # tracked globally; counts per occurrence below
                continue
            pairs.append({
                "source_group_id": src["source_group_id"],
                "source_record_id": src["source_record_id"],
                "split": split_name,
                "relation": src["relation_nfkc"],
                "question": src["question_nfkc"],
                "r_o": src["canonical_answer_nfkc"],
                "r_s": r_s,
                "donor_group_id": donor["source_group_id"],
                "donor_record_id": donor["source_record_id"],
                "c_o": c_o,
                "c_s": c_s,
            })
            audit_rows.append({"source_group_id": src["source_group_id"], "split": split_name,
                               "relation": src["relation_nfkc"], "donor_group_id": donor["source_group_id"],
                               "r_o": src["canonical_answer_nfkc"], "r_s": r_s,
                               "answer_norm_equal": src["answer_norm"] == donor["answer_norm"]})

print(f"R7 excluded: {r7_excluded}, R8 excluded: {r8_excluded}")
print("retained pairs:", len(pairs))

# extend funnel
funnel = json.load(open(R / "scripts" / "_funnel_r1_6.json"))
funnel["R7"] = r7_excluded
funnel["R8"] = r8_excluded
funnel["surviving_final"] = len(pairs)
with open(R / "filter_funnel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["stage", "excluded", "remaining"])
    cum = funnel["initial"]
    for k in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"):
        w.writerow([k, funnel[k], cum - funnel[k]])
        cum -= funnel[k]
    w.writerow(["surviving_r6", 0, funnel["surviving_r6"]])
    w.writerow(["surviving_final", 0, len(pairs)])

# write external_swap_pairs.jsonl (rows keep raw answers; candidate rendering in Phase 3)
with open(R / "external_swap_pairs.jsonl", "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

# donor audit
with open(R / "donor_assignment_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
    w.writeheader()
    w.writerows(audit_rows)

json.dump({"r7_excluded": r7_excluded, "r8_excluded": r8_excluded, "retained": len(pairs)},
          open(R / "scripts" / "_phase2_stats.json", "w"), indent=2)
print("Phase 2 OK")
