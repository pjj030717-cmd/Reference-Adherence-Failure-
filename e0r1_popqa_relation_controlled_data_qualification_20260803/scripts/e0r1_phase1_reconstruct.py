#!/usr/bin/env python3
"""E0-R1 Phase 1: reconstruct the full E0 pipeline (verbatim) in the E0-R1 directory.

Source: E0/source/test.tsv (SHA256 already verified in E0). Re-runs:
  - Phase 0: schema + source_group_id
  - Phase 1: R1-R6 (Qwen tokenizer only, no AutoModel)
  - Phase 2: split (seed 20260816, 60/20/20), R7/R8, donor (per-group RNG "20260816|sgid")
Then verifies the reconstructed artifacts match E0's artifacts exactly.

Writes: source_access_audit.md, source_data_contract.md, filter_funnel.csv,
        fixed_split_indices.json, eligible_source_groups.jsonl,
        external_swap_pairs.jsonl, donor_selection_audit.csv, _records.jsonl,
        _after_r6.jsonl, _pairs_shared.json (shared stash for phase 2)
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

from transformers import AutoTokenizer

R = REPO_ROOT / "e0r1_popqa_relation_controlled_data_qualification_20260803"
E0 = REPO_ROOT / "e0_popqa_relation_controlled_data_qualification_20260803"
T0 = "The answer is <answer>."
SEP = "\x00"
SEED_SPLIT = 20260816
EXP_SHA = {"test.tsv": "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b",
           "README.md": "bb04b56bc87a3b2865cc2e2a1649ba6c766a7a44dcba5a53170fbfc72c0da9f0"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", (s or "")).split())


# ---- source files copied reference (read from E0, do not redownload) ----
for fn, exp in EXP_SHA.items():
    got = sha256_file(E0 / "source" / fn)
    if got != exp:
        print("STOP: source hash mismatch", fn, got)
        sys.exit(1)
    print("OK source hash", fn)

# ---- Phase 0: schema + source_group_id ----
rows = []
with open(E0 / "source" / "test.tsv", encoding="utf-8") as fh:
    rd = csv.DictReader(fh, delimiter="\t")
    for r in rd:
        rows.append(r)
print("rows:", len(rows))
assert len(rows) == 14267

records = []
seen = set()
for r in rows:
    q_n = unicodedata.normalize("NFKC", r["question"])
    o_n = unicodedata.normalize("NFKC", r["obj"])
    p_n = unicodedata.normalize("NFKC", r["prop"])
    id_s = str(r["id"])
    raw = q_n + SEP + o_n + SEP + p_n + SEP + id_s
    sgid = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert sgid not in seen
    seen.add(sgid)
    records.append({
        "source_record_id": id_s, "source_group_id": sgid,
        "question_nfkc": q_n, "canonical_answer_nfkc": o_n, "relation_nfkc": p_n,
        "question_norm": norm(r["question"]), "canonical_answer_norm": norm(r["obj"]),
        "relation_norm": norm(r["prop"]),
    })
print("records:", len(records))

# ---- Phase 1: R1-R6 ----
tok = AutoTokenizer.from_pretrained(os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"))
print("tokenizer loaded (no AutoModel)")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\n\r]")
funnel = {"initial": len(records), "R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0, "R6": 0}
surviving = []
for rec in records:
    q_n, a_n, rel_n = rec["question_nfkc"], rec["canonical_answer_nfkc"], rec["relation_nfkc"]
    if not q_n.strip() or not a_n.strip() or not rel_n.strip():
        funnel["R1"] += 1
        continue
    if CTRL.search(a_n):
        funnel["R2"] += 1
        continue
    a_nfc = unicodedata.normalize("NFKC", a_n)
    if not (1 <= len(a_nfc) <= 80):
        funnel["R3"] += 1
        continue
    n_ans = len(tok.encode(a_nfc, add_special_tokens=False))
    if not (1 <= n_ans <= 16):
        funnel["R4"] += 1
        continue
    n_q = len(tok.encode(q_n, add_special_tokens=False))
    if n_q > 192:
        funnel["R5"] += 1
        continue
    if a_nfc in q_n:
        funnel["R6"] += 1
        continue
    rec["answer_nfkc"] = a_nfc
    rec["answer_norm"] = norm(a_n)
    rec["question_norm"] = norm(q_n)
    rec["relation_norm"] = norm(rel_n)
    rec["answer_n_tok"] = n_ans
    rec["question_n_tok"] = n_q
    surviving.append(rec)
funnel["surviving_r6"] = len(surviving)
print("funnel R1-R6:", funnel)

with open(R / "scripts" / "_records.jsonl", "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
with open(R / "scripts" / "_after_r6.jsonl", "w", encoding="utf-8") as f:
    for rec in surviving:
        f.write(json.dumps(rec) + "\n")

# ---- Phase 2: split + R7/R8 + donor ----
records_sorted = sorted(surviving, key=lambda r: r["source_group_id"])
rng = random.Random(SEED_SPLIT)
rng.shuffle(records_sorted)
n = len(records_sorted)
n_train = int(n * 0.6)
n_dev = int(n * 0.2)
train = records_sorted[:n_train]
dev = records_sorted[n_train:n_train + n_dev]
final_reserve = records_sorted[n_train + n_dev:]
print(f"split: train={len(train)} dev={len(dev)} final={len(final_reserve)}")

sorted_ids = [r["source_group_id"] for r in records_sorted]
split_of_idx = {}
for i, r in enumerate(records_sorted):
    split_of_idx[i] = "train" if i < n_train else ("dev" if i < n_train + n_dev else "final_reserve")
json.dump({"seed": SEED_SPLIT, "shuffle_applied": True,
           "counts": {"train": len(train), "dev": len(dev), "final_reserve": len(final_reserve)},
           "n_total": n,
           "index_to_split": {str(k): v for k, v in split_of_idx.items()},
           "sorted_source_group_ids": sorted_ids},
          open(R / "fixed_split_indices.json", "w"), indent=2)

r7 = 0
r8 = 0
pairs = []
audit = []
for split_name, split_rows in (("train", train), ("dev", dev), ("final_reserve", final_reserve)):
    by_rel = defaultdict(list)
    for rec in split_rows:
        by_rel[rec["relation_nfkc"]].append(rec)
    for rel, recs in by_rel.items():
        if len(recs) < 2:
            r7 += len(recs)
            continue
        for src in sorted(recs, key=lambda r: r["source_group_id"]):
            cands = [c for c in recs if c["source_group_id"] != src["source_group_id"]
                     and c["answer_norm"] != src["answer_norm"]]
            cands.sort(key=lambda c: c["source_group_id"])
            if not cands:
                r8 += 1
                continue
            grng = random.Random(f"{SEED_SPLIT}|{src['source_group_id']}")
            donor = grng.choice(cands)
            r_s = donor["canonical_answer_nfkc"]
            c_o = T0.replace("<answer>", src["canonical_answer_nfkc"])
            c_s = T0.replace("<answer>", r_s)
            if unicodedata.normalize("NFKC", c_o) == unicodedata.normalize("NFKC", c_s):
                continue  # R9, should be 0
            pairs.append({"source_group_id": src["source_group_id"],
                          "source_record_id": src["source_record_id"], "split": split_name,
                          "relation": src["relation_nfkc"], "question": src["question_nfkc"],
                          "r_o": src["canonical_answer_nfkc"], "r_s": r_s,
                          "donor_group_id": donor["source_group_id"],
                          "donor_record_id": donor["source_record_id"],
                          "c_o": c_o, "c_s": c_s})
            audit.append({"source_group_id": src["source_group_id"], "split": split_name,
                          "relation": src["relation_nfkc"], "donor_group_id": donor["source_group_id"],
                          "r_o": src["canonical_answer_nfkc"], "r_s": r_s,
                          "answer_norm_equal": src["answer_norm"] == donor["answer_norm"]})
print(f"R7={r7} R8={r8} retained={len(pairs)}")
funnel["R7"] = r7
funnel["R8"] = r8
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
json.dump(funnel, open(R / "scripts" / "_funnel.json", "w"), indent=2)

with open(R / "external_swap_pairs.jsonl", "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
with open(R / "donor_selection_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
    w.writeheader()
    w.writerows(audit)

# ---- verify identical to E0 ----
e0_pairs = {}
for line in open(E0 / "external_swap_pairs.jsonl", encoding="utf-8"):
    p = json.loads(line)
    e0_pairs[p["source_group_id"]] = p
mism = 0
for p in pairs:
    e = e0_pairs.get(p["source_group_id"])
    if e is None or e["r_s"] != p["r_s"] or e["donor_group_id"] != p["donor_group_id"] or e["split"] != p["split"]:
        mism += 1
print("pair mismatch vs E0:", mism, "/", len(pairs))
if mism or len(pairs) != len(e0_pairs):
    print("STOP: reconstruction not identical to E0")
    sys.exit(1)

# eligible_source_groups.jsonl
with open(R / "eligible_source_groups.jsonl", "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps({"source_group_id": p["source_group_id"],
                            "source_record_id": p["source_record_id"], "split": p["split"],
                            "relation": p["relation"], "question": p["question"],
                            "canonical_answer": p["r_o"]}) + "\n")

# source access audit (inherited)
(R / "source_access_audit.md").write_text(
    (E0 / "source_access_audit.md").read_text(encoding="utf-8").replace(
        "# source_access_audit.md", "# source_access_audit.md\n\n> 由 E0 逐字继承（E0-R1 不重新下载；原始文件仍在 E0/source/，SHA256 已复核）。"), encoding="utf-8")

# stash pairs shared to phase2
json.dump([{"source_group_id": p["source_group_id"], "split": p["split"], "relation": p["relation"],
            "question": p["question"], "r_o": p["r_o"], "r_s": p["r_s"]} for p in pairs],
          open(R / "scripts" / "_pairs_shared.json", "w"))
print("E0-R1 Phase 1 OK; reconstruction identical to E0")
