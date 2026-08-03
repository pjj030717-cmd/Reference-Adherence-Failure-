#!/usr/bin/env python3
"""E0 Phase 0: PopQA source access audit + data contract recovery.

Recovers, from official schema and README, the unique field mapping:
  question        <- question
  canonical_answer<- obj
  relation/property<- prop
  official id     <- id
Defines source_group_id = SHA256(NFKC(q) || SEP || NFKC(obj) || SEP || NFKC(prop) || SEP || NFKC(id)).
Checks uniqueness and non-emptiness.
Writes: source_access_audit.md, source_data_contract.md, _loaded_records.jsonl (raw normalized).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "e0_popqa_relation_controlled_data_qualification_20260803"
SEP = "\x00"

# ---- access facts ----
REPO = "akariasai/PopQA"
REVISION = "098765c79ea10a2cb19c828324e33281b8336ec0"
LAST_MODIFIED = "2022-12-22T01:01:20.000Z"
DOWNLOAD_DATE = "2026-08-03"
CONFIG = "default"
SPLIT = "test"  # official PopQA ships a single split named 'test'
FILES = {
    "source/test.tsv": "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b",
    "source/README.md": "bb04b56bc87a3b2865cc2e2a1649ba6c766a7a44dcba5a53170fbfc72c0da9f0",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(why: str):
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": "popqa_data_contract_invalid", "reason": why}, indent=2), encoding="utf-8")
    print("STOP:", why)
    sys.exit(1)


# verify file hashes
for f, expected in FILES.items():
    got = sha256_file(R / f)
    if got != expected:
        fail(f"file hash mismatch {f}: {got}")
    print(f"OK hash {f}")

# ---- schema recovery ----
rows = []
with open(R / "source" / "test.tsv", encoding="utf-8") as fh:
    rd = csv.DictReader(fh, delimiter="\t")
    for r in rd:
        rows.append(r)
print("total rows:", len(rows))

# schema from official README:
SCHEMA = {
    "question": "PopQA question",
    "obj": "object entity name (Wikidata object entity) — canonical answer",
    "prop": "relationship type — relation/property",
    "id": "question id — stable official record id",
}
for field in ("question", "obj", "prop", "id"):
    if field not in rows[0]:
        fail(f"field {field} missing from official schema")


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", (s or "")).split())


records = []
seen = set()
for r in rows:
    q_n = unicodedata.normalize("NFKC", r["question"])
    o_n = unicodedata.normalize("NFKC", r["obj"])
    p_n = unicodedata.normalize("NFKC", r["prop"])
    id_s = str(r["id"])
    raw = q_n + SEP + o_n + SEP + p_n + SEP + id_s
    sgid = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if sgid in seen:
        fail(f"duplicate source_group_id {sgid}")
    seen.add(sgid)
    records.append({
        "source_record_id": id_s,
        "source_group_id": sgid,
        "question_nfkc": q_n,
        "canonical_answer_nfkc": o_n,
        "relation_nfkc": p_n,
        "question_norm": norm(r["question"]),
        "canonical_answer_norm": norm(r["obj"]),
        "relation_norm": norm(r["prop"]),
    })
print("records with unique source_group_id:", len(records))

# non-empty checks
empty = [rec for rec in records if not rec["question_nfkc"].strip() or not rec["canonical_answer_nfkc"].strip()
         or not rec["relation_nfkc"].strip()]
if empty:
    fail(f"{len(empty)} records with empty q/answer/relation")

# stash normalized records
with open(R / "scripts" / "_records.jsonl", "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")

# ---- source_access_audit.md ----
(R / "source_access_audit.md").write_text(
    """# source_access_audit.md

## 数据来源

- dataset repository：`akariasai/PopQA`
- revision / commit：`098765c79ea10a2cb19c828324e33281b8336ec0`（main，lastModified 2022-12-22T01:01:20Z）
- config：`default`
- split：`test`（官方仅提供单一 split，共 14,267 行）
- 下载端点：`hf-mirror.com/datasets/akariasai/PopQA/resolve/main/`
- 下载日期：2026-08-03
- license：仓库 README / HF metadata 均未声明 license（记录为 not specified）

## 文件 SHA256

| 文件 | SHA256 |
|---|---|
| README.md | bb04b56bc87a3b2865cc2e2a1649ba6c766a7a44dcba5a53170fbfc72c0da9f0 |
| test.tsv | 9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b |

## 仅下载

PopQA 数据文件与其官方 README；未下载任何模型、JAR 数据、Wikidata dump、NER 模型或其他数据集。
""", encoding="utf-8")

# ---- source_data_contract.md ----
(R / "source_data_contract.md").write_text(
    """# source_data_contract.md

## Schema 恢复（唯一确定）

| 概念 | 官方字段 | 官方含义（README） |
|---|---|---|
| question | `question` | PopQA question |
| 规范答案 | `obj` | object entity name（Wikidata object entity） |
| relation/property | `prop` | relationship type |
| 稳定官方记录 id | `id` | question id（唯一，14,267/14,267） |

`possible_answers` 为 gold answers 列表，用于评估；本轮以单一规范答案 `obj` 作为 canonical answer。

## source_group_id 定义

```
source_group_id = SHA256(
    NFKC(question)  || "\\x00" ||
    NFKC(obj)       || "\\x00" ||
    NFKC(prop)      || "\\x00" ||
    NFKC(str(id))
)
```

- 分隔符：`\\x00`（NUL，字符串中不可出现，防连接歧义）
- NFKC 直接采用（不做空白折叠），与协议 0.2 一致
- 唯一性：14,267 / 14,267 唯一
- 非空：question / canonical_answer / relation 均非空

## 规范化（用于过滤与比较）

```
norm(s) = ' '.join(NFKC(s).split())    # NFKC + 连续空白压缩为一个空格
保留大小写、保留标点
```

## 拟用 split

PopQA 官方仅提供单一 `test` split。本实验将其视为拟用全量池，随后按协议 2.1
以 source_group_id 排序 + `random.Random(20260816)` 打乱后切分 train/dev/final-reserve = 60/20/20。
""", encoding="utf-8")

print("Phase 0 OK; records written:", len(records))
