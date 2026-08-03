#!/usr/bin/env python3
"""E0 Phase 1: pre-registered mechanical filter R1-R6 (record-level) + tokenizer audit.

Only Qwen tokenizer is loaded (AutoTokenizer), NO AutoModel / weights / forward.
R1: question/answer/relation empty
R2: canonical_answer contains newline/control chars
R3: NFKC(answer) len <1 or >80
R4: answer token count (Qwen tok) <1 or >16
R5: question token count (Qwen tok) >192
R6: NFKC(answer) appears in NFKC(question), case-sensitive

Writes filter_funnel.csv (incremental) and _after_r6.jsonl.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

from transformers import AutoTokenizer

R = REPO_ROOT / "e0_popqa_relation_controlled_data_qualification_20260803"
MODEL_TOK = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", (s or "")).split())


def fail(why: str):
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": "popqa_data_contract_invalid", "reason": why}, indent=2), encoding="utf-8")
    print("STOP:", why)
    sys.exit(1)


records = [json.loads(line) for line in open(R / "scripts" / "_records.jsonl", encoding="utf-8")]
print("records:", len(records))

# tokenizer (tokenizer only; no model weights)
tok = AutoTokenizer.from_pretrained(MODEL_TOK)
print("tokenizer loaded:", type(tok).__name__)

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\n\r]")

# funnel counters
funnel = {"initial": len(records), "R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0, "R6": 0}
surviving = []

for rec in records:
    ok = True
    q_n, a_n, rel_n = rec["question_nfkc"], rec["canonical_answer_nfkc"], rec["relation_nfkc"]
    # R1
    if not q_n.strip() or not a_n.strip() or not rel_n.strip():
        funnel["R1"] += 1
        continue
    # R2
    if CTRL.search(a_n):
        funnel["R2"] += 1
        continue
    # R3
    a_nfc = unicodedata.normalize("NFKC", a_n)
    if not (1 <= len(a_nfc) <= 80):
        funnel["R3"] += 1
        continue
    # R4
    n_ans_tok = len(tok.encode(a_nfc, add_special_tokens=False))
    if not (1 <= n_ans_tok <= 16):
        funnel["R4"] += 1
        continue
    # R5
    n_q_tok = len(tok.encode(q_n, add_special_tokens=False))
    if n_q_tok > 192:
        funnel["R5"] += 1
        continue
    # R6
    if a_nfc in q_n:
        funnel["R6"] += 1
        continue
    rec["answer_nfkc"] = a_nfc
    rec["answer_norm"] = norm(a_n)
    rec["question_norm"] = norm(q_n)
    rec["relation_norm"] = norm(rel_n)
    rec["answer_n_tok"] = n_ans_tok
    rec["question_n_tok"] = n_q_tok
    surviving.append(rec)

funnel["surviving_r6"] = len(surviving)
print("funnel:", funnel)
print("surviving R1-R6:", len(surviving))

# write funnel CSV
with open(R / "filter_funnel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["stage", "excluded", "remaining"])
    cum = funnel["initial"]
    for k in ("R1", "R2", "R3", "R4", "R5", "R6"):
        w.writerow([k, funnel[k], cum - funnel[k]])
        cum -= funnel[k]
    w.writerow(["surviving_r6", 0, len(surviving)])

# stash
with open(R / "scripts" / "_after_r6.jsonl", "w", encoding="utf-8") as f:
    for rec in surviving:
        f.write(json.dumps(rec) + "\n")
json.dump(funnel, open(R / "scripts" / "_funnel_r1_6.json", "w"), indent=2)
print("Phase 1 OK")
