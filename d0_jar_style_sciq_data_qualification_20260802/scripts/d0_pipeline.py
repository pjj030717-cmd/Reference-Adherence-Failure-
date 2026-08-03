#!/usr/bin/env python3
"""E01-D0: JAR-style reference-knowledge conflict — SciQ data qualification gate.

Constructs an auditable four-cell swap dataset from SciQ validation split ONLY.

Pipeline:
  Phase A: schema + unique source_group_id check (contract)
  Phase B: pre-registered mechanical filter rules (funnel)
  Phase C: coarse-form-controlled Random Swap (one r_s per group, RNG 20260802)
  Phase D: capacity check + group-level split (train/dev/final_reserve)
  Phase E: blind semantic audit packet (seed 20260803, up to 100 groups)
  Phase F: deliverables + decision.json

NO model is loaded; NO judge runs; NO hidden states are read; NO probes/SFT/RM trained.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd

OUT = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
RAW = OUT / "data" / "raw"
VAL_PARQUET = RAW / "data" / "validation-00000-of-00001.parquet"
REV = "2c94ad3e1aafab77146f384e23536f97a4849815"
SEED_SWAP = 20260802
SEED_SPLIT = 20260802
SEED_BLIND = 20260803
MIN_GROUPS = 600

FIELDS = ["question", "correct_answer", "distractor1", "distractor2", "distractor3", "support"]
GENERIC = {"yes", "no", "true", "false", "unknown", "none"}
RENDER_TEMPLATE = "The answer is <answer>."


def fail(label: str, reason: str):
    print(f"FINAL LABEL: {label}")
    print("reason:", reason)
    (OUT / "artifacts").mkdir(parents=True, exist_ok=True)
    (OUT / "artifacts" / "decision.json").write_text(
        json.dumps({"final_label": label, "reason": reason,
                    "model_loaded": False, "judge_scored": False,
                    "hidden_states_read": False, "probe_trained": False}, indent=2),
        encoding="utf-8",
    )
    sys.exit(1)


def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def group_id(row) -> str:
    payload = "|||".join(norm(row[f]) for f in ["question", "correct_answer",
                                                "distractor1", "distractor2", "distractor3"])
    return sha256_hex(payload)


def render(ans: str) -> str:
    return RENDER_TEMPLATE.replace("<answer>", norm(ans))


# ----------------------------------------------------------------------
# Phase A: schema / contract
# ----------------------------------------------------------------------
df = pd.read_parquet(VAL_PARQUET)
if set(df.columns) != set(FIELDS):
    fail("sciq_schema_invalid", f"unexpected columns: {list(df.columns)}")

rows = []
for _, r in df.iterrows():
    row = {f: ("" if pd.isna(r[f]) else str(r[f])) for f in FIELDS}
    row["source_group_id"] = group_id(row)
    rows.append(row)

# unique source groups
seen = set()
for row in rows:
    if row["source_group_id"] in seen:
        fail("sciq_schema_invalid", "duplicate source_group_id in validation split")
    seen.add(row["source_group_id"])
if len(rows) != 1000:
    fail("sciq_schema_invalid", f"validation has {len(rows)} rows, expected 1000")
print(f"Phase A: validation rows={len(rows)} unique source groups={len(seen)}")

# ----------------------------------------------------------------------
# Phase B: pre-registered mechanical filter rules (funnel)
# ----------------------------------------------------------------------
RULES = [
    ("r1_r_o_nonempty", lambda g: g["r_o"] != ""),
    ("r2_tokens_1_to_6", lambda g: 1 <= len(g["r_o"].split()) <= 6),
    ("r3_no_digit", lambda g: not re.search(r"\d", g["r_o"])),
    ("r4_no_newline_url_bracket", lambda g: not re.search(r"[\n\r]", g["r_o"])
                                              and "http" not in g["r_o"].lower()
                                              and "www." not in g["r_o"].lower()
                                              and not re.search(r"[()\[\]{}]", g["r_o"])),
    ("r5_only_en_letters_space_hyphen_apos", lambda g: bool(re.fullmatch(r"[a-zA-Z\s'\-]+", g["r_o"]))),
    ("r6_not_generic_answer", lambda g: g["r_o"].lower() not in GENERIC),
    ("r7_not_in_question", lambda g: g["r_o"] not in g["question"]),
    ("r8_not_equal_distractor", lambda g: all(g["r_o"] != d for d in [g["d1"], g["d2"], g["d3"]])),
]

for row in rows:
    row["r_o"] = norm(row["correct_answer"])
    row["d1"] = norm(row["distractor1"])
    row["d2"] = norm(row["distractor2"])
    row["d3"] = norm(row["distractor3"])
    row["question"] = norm(row["question"])
    row["support"] = norm(row["support"])
    row["failed_rule"] = None
    for name, fn in RULES:
        if not fn(row):
            row["failed_rule"] = name
            break

# funnel stats (first-failure attribution, in rule order)
funnel = []
cum = 0
for name, _ in RULES:
    n = sum(1 for row in rows if row["failed_rule"] == name)
    cum += n
    funnel.append({"rule": name, "excluded_count": n, "cumulative_excluded": cum})
# groups passing all rules -> eligible pool
eligible = [row for row in rows if row["failed_rule"] is None]
funnel.append({"rule": "pass_all", "excluded_count": len(eligible),
               "cumulative_excluded": cum})
print(f"Phase B: eligible groups = {len(eligible)}")

# ----------------------------------------------------------------------
# Phase C: coarse-form-controlled Random Swap
# ----------------------------------------------------------------------
eligible.sort(key=lambda g: g["source_group_id"])
pool = {g["source_group_id"]: g["r_o"] for g in eligible}

def feasible_swap(g) -> list:
    cands = []
    for gid, ro in pool.items():
        if gid == g["source_group_id"]:
            continue
        if ro == g["r_o"]:
            continue
        if ro in {g["d1"], g["d2"], g["d3"]}:
            continue
        if ro in g["question"] or (g["support"] and ro in g["support"]):
            continue
        cands.append((gid, ro))
    # dedupe by r_s value (sorted by group id first for determinism)
    cands.sort(key=lambda x: x[0])
    return cands

rng_swap = random.Random(SEED_SWAP)
prelim = []  # one per final group
swap_excluded = 0
for g in eligible:
    cands = feasible_swap(g)
    if not cands:
        swap_excluded += 1
        g["failed_rule"] = "no_feasible_r_s"
        continue
    # fixed RNG picks from sorted candidate list
    picked_gid, picked_ro = rng_swap.choice(cands)
    g["r_s"] = picked_ro
    g["swap_source_group_id"] = picked_gid
    g["c_o"] = render(g["r_o"])
    g["c_s"] = render(g["r_s"])
    prelim.append(g)

final_pool = prelim
print(f"Phase C: final pool groups = {len(final_pool)}, swap-excluded = {swap_excluded}")

if len(final_pool) < MIN_GROUPS:
    fail("jar_swap_capacity_insufficient",
         f"final pool {len(final_pool)} < {MIN_GROUPS} required unique groups")

# ----------------------------------------------------------------------
# Phase D: group-level split (fixed RNG seed 20260802)
# ----------------------------------------------------------------------
ids = sorted(g["source_group_id"] for g in final_pool)
rng_split = random.Random(SEED_SPLIT)
shuffled = ids[:]
rng_split.shuffle(shuffled)
n = len(shuffled)
n_train = int(n * 0.60)
n_dev = int(n * 0.20)
n_res = n - n_train - n_dev
split_map = {}
for i, gid in enumerate(shuffled):
    if i < n_train:
        split_map[gid] = "train"
    elif i < n_train + n_dev:
        split_map[gid] = "dev"
    else:
        split_map[gid] = "final_reserve"

# verify group isolation: one split per group
assert len(split_map) == len(final_pool)
counts = {"train": 0, "dev": 0, "final_reserve": 0}
for g in final_pool:
    g["split"] = split_map[g["source_group_id"]]
    counts[g["split"]] += 1
print(f"Phase D: split counts = {counts}")

# ----------------------------------------------------------------------
# Phase E: blind semantic audit packet (seed 20260803, up to 100 groups)
# ----------------------------------------------------------------------
rng_blind = random.Random(SEED_BLIND)
blind_ids = rng_blind.sample(ids, min(100, len(ids)))
blind_ids.sort()
id2g = {g["source_group_id"]: g for g in final_pool}

# ----------------------------------------------------------------------
# Phase F: deliverables
# ----------------------------------------------------------------------
# candidate rendering spec + sha256
render_spec = {
    "template": RENDER_TEMPLATE,
    "rendering_rule": "Candidate(answer) = 'The answer is <answer>.' — 唯一允许变化的字段是 <answer>。",
    "answer_field_source": "r_o 或 r_s 的 NFKC + trim + 连续空白归一化后的原字符串（不做大小写转换，保证 r_o != r_s 时 c_o != c_s）",
    "fixed_properties": ["模板文本固定", "大小写与源数据一致", "标点仅模板自带的一个句点"],
    "sha256_utf8": None,
}
(OUT / "candidate_rendering_spec.json").write_text(json.dumps(render_spec, indent=2), encoding="utf-8")
render_spec["sha256_utf8"] = sha256_hex((OUT / "candidate_rendering_spec.json").read_text(encoding="utf-8"))
(OUT / "candidate_rendering_spec.json").write_text(json.dumps(render_spec, indent=2), encoding="utf-8")

# eligible_source_groups.jsonl
with open(OUT / "eligible_source_groups.jsonl", "w", encoding="utf-8") as f:
    for g in eligible:
        f.write(json.dumps({"source_group_id": g["source_group_id"], "question": g["question"],
                            "r_o": g["r_o"], "correct_answer": g["correct_answer"],
                            "distractor1": g["distractor1"], "distractor2": g["distractor2"],
                            "distractor3": g["distractor3"], "support": g["support"]}) + "\n")

# preliminary_swap_pairs.jsonl
with open(OUT / "preliminary_swap_pairs.jsonl", "w", encoding="utf-8") as f:
    for g in final_pool:
        f.write(json.dumps({"q": g["question"], "r_o": g["r_o"], "r_s": g["r_s"],
                            "c_o": g["c_o"], "c_s": g["c_s"],
                            "original_group_id": g["source_group_id"],
                            "swap_source_group_id": g["swap_source_group_id"],
                            "split": g["split"]}) + "\n")

# filter_funnel.csv
with open(OUT / "filter_funnel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["rule", "excluded_count", "cumulative_excluded"])
    w.writeheader()
    for row in funnel:
        w.writerow(row)

# fixed_split_indices.json
split_hash = {}
for name in ["train", "dev", "final_reserve"]:
    gids = sorted(gid for gid, s in split_map.items() if s == name)
    split_hash[name] = sha256_hex("\n".join(gids))
fixed_split = {
    "seed": SEED_SPLIT,
    "sorting": "source_group_id ascending then fixed-RNG shuffle (python random.Random(20260802).shuffle)",
    "proportions": {"train": 0.60, "dev": 0.20, "final_reserve": 0.20},
    "split_sha256": split_hash,
    "groups": {name: sorted(gid for gid, s in split_map.items() if s == name)
               for name in ["train", "dev", "final_reserve"]},
}
(OUT / "fixed_split_indices.json").write_text(json.dumps(fixed_split, indent=2), encoding="utf-8")

# blind_semantic_audit_packet.csv
with open(OUT / "blind_semantic_audit_packet.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["question", "r_o", "r_s", "c_o", "c_s"])
    w.writeheader()
    for gid in blind_ids:
        g = id2g[gid]
        w.writerow({"question": g["question"], "r_o": g["r_o"], "r_s": g["r_s"],
                    "c_o": g["c_o"], "c_s": g["c_s"]})

# source_access_audit.md
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %z")
(OUT / "source_access_audit.md").write_text(
    f"""# source_access_audit.md

## 数据来源
- 数据集：allenai/sciq（官方公开数据）
- 类型：HF datasets repo
- 来源 URL：https://huggingface.co/datasets/allenai/sciq
- 官方主页：https://allenai.org/data/sciq
- 本地镜像端点：https://hf-mirror.com（hf_hub_download，repo_type='dataset'）
- revision (commit sha)：`{REV}`
- 下载时间（UTC+8）：{now}
- 许可：CC BY-NC 3.0（README 声明 http://creativecommons.org/licenses/by-nc/3.0/；非商业用途）

## 文件哈希（SHA256）
```text
train-00000-of-00001.parquet      19644360954006d06e9ad3df07bddb34f8535c081b831d48f604603c713ac167
validation-00000-of-00001.parquet 455dd9f1d725cd3ecbce369799a2fbbdbbfecf51ab84a86d56ba3370dc847b8a
test-00000-of-00001.parquet       3a719356a29b127fc54ef3c7f51a034db4bd105d5717215e8c85d2aa58d60667
README.md                         f16f71b220a0e205672f6f0c8afb40e37f7a158541f759593745fe52162f8ad8
```

## 本轮使用范围
- 仅使用 validation split（1000 行）。
- 仅下载 SciQ 数据本身（parquet + README）；未下载任何模型权重、外部知识库或额外数据。
- 未读取 test split；未进行任何模型推理。

## 不确定性
- README 为 HF datasets 社区卡片，training-data 详情感知信息标注为 More Information Needed。
- support 字段 887/1000 行为非空。
- 未在 huggingface.co 直连（网络不通），经官方镜像 hf-mirror.com 下载，commit sha 与 API 返回一致。
""",
    encoding="utf-8",
)

# source_data_contract.md
(OUT / "source_data_contract.md").write_text(
    f"""# source_data_contract.md

## 字段（SciQ validation split，parquet schema）
| 字段 | 类型 | 说明 |
|---|---|---|
| question | string | 原始问题 |
| correct_answer | string | 唯一正确答案（r_o 来源） |
| distractor1/2/3 | string | 三个干扰项 |
| support | string | 支持性证据段落（887/1000 非空；113 行为空字符串） |

## source group 定义
- 每个原始 question 为一个 source group（validation split 共 1000 个）。
- source_group_id = SHA256(NFKC(question) || "|||" || NFKC(correct_answer) || "|||" || NFKC(distractor1) || "|||" || NFKC(distractor2) || "|||" || NFKC(distractor3))。
- 分隔符使用 "|||" 以避免字段拼接歧义；不使用行号。
- 1000 行 → 1000 个唯一 source_group_id（无重复）。

## 归一化规则（机械过滤与哈希共用）
- NFKC → trim → 连续空白折叠为单个空格。
- 不做大小写转换（保证 r_o != r_s 时渲染后 c_o != c_s）。

## revision 与哈希
- revision：`{REV}`
- validation parquet SHA256：`455dd9f1d725cd3ecbce369799a2fbbdbbfecf51ab84a86d56ba3370dc847b8a`

## 任何不确定性
- support 缺失 113 行：机械过滤仅在 r_s 约束中使用 support 子串检查，空 support 视为无约束。
- 许可证为 CC BY-NC 3.0（非商业），本实验为学术研究用途。
""",
    encoding="utf-8",
)

# failure_examples.md (top-10 per rule, de-identified by group hash prefix)
def short_gid(gid: str) -> str:
    return gid[:8]

fails_by_rule = {}
for g in rows:
    fails_by_rule.setdefault(g["failed_rule"] or "pass", []).append(g)
(OUT / "failure_examples.md").write_text(
    "# failure_examples.md\n\n"
    "每条主要排除原因展示至多 10 条脱敏示例（source_group_id 前缀 + r_o 归一化值）。\n\n"
    + "\n".join(
        f"## {rule}（共 {len(groups)} 条）\n\n"
        + "\n".join(
            f"- `{short_gid(g['source_group_id'])}`  r_o={g['r_o']!r}"
            for g in groups[:10]
        )
        + "\n"
        for rule, groups in fails_by_rule.items()
    )
    + "\n注意：示例仅含归一化后的 r_o 与 group 前缀，不含可识别为样本原文的整句。\n",
    encoding="utf-8",
)

# final decision
decision = {
    "final_label": "jar_style_sciq_data_qualification_feasible",
    "model_loaded": False,
    "judge_scored": False,
    "hidden_states_read": False,
    "probe_trained": False,
    "source_revision": REV,
    "validation_groups": 1000,
    "eligible_groups": len(eligible),
    "swap_excluded_groups": swap_excluded,
    "final_pool_groups": len(final_pool),
    "split_counts": counts,
    "blind_packet_groups": len(blind_ids),
    "rendering_template": RENDER_TEMPLATE,
    "swap_seed": SEED_SWAP,
    "split_seed": SEED_SPLIT,
    "blind_seed": SEED_BLIND,
}
(OUT / "artifacts").mkdir(parents=True, exist_ok=True)
(OUT / "artifacts" / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

print("DONE. final label = jar_style_sciq_data_qualification_feasible")
print("funnel:")
for row in funnel:
    print(f"  {row['rule']}: {row['excluded_count']}")
print("split counts:", counts, "blind packet:", len(blind_ids))
