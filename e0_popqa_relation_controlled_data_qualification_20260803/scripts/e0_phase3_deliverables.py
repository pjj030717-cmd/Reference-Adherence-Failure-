#!/usr/bin/env python3
"""E0 Phase 3: template contract + capacity check + deliverables (both outcomes).

Mechanical outputs always produced: candidate_template_contract_audit.csv,
eligible_source_groups.jsonl, relation_distribution_by_split.csv,
relation_controlled_swap_spec.json/.sha256, failure_examples.md.

If capacity gates pass: also blind_candidate_contract_packet.csv and
final_label = popqa_relation_controlled_external_data_feasible.
If capacity fails: final_label = popqa_relation_swap_capacity_insufficient,
blind packet NOT produced, final_report honestly reports the failure.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "e0_popqa_relation_controlled_data_qualification_20260803"
D1RA = REPO_ROOT / "d1ra_candidate_template_provenance_diversity_audit_20260803"

EXPECTED = {
    "T0": "The answer is <answer>.",
    "T1": "For this question, the answer is <answer>.",
    "T2": "The response is <answer>.",
}
EXPECTED_SHA = {
    "T0": "c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc",
    "T1": "d325f862ad174533fe38c193744bbebd30b23e2ec72905a173c2b2eaed8fc078",
    "T2": "5fb1b5ed1ba1cb158981aea1673d936dcf88ff91b1423c6796031886de47df24",
}

canon = json.loads((D1RA / "canonical_candidate_templates.json").read_text(encoding="utf-8"))
TEMPLATES = {}
for k in ("T0", "T1", "T2"):
    t = canon[k]["template"]
    if t != EXPECTED[k] or hashlib.sha256(t.encode("utf-8")).hexdigest() != EXPECTED_SHA[k]:
        sys.exit(f"STOP: {k} template mismatch")
    TEMPLATES[k] = t
print("templates verified")

SCORE_WORDS = ["correct", "valid", "accept", "reject"]
for k, t in TEMPLATES.items():
    fixed = t.replace("<answer>", "")
    assert "reference" not in fixed.lower(), f"{k} reference mention"
    for w in SCORE_WORDS:
        assert w not in fixed.lower(), f"{k} scoring word {w}"
    assert t.count("<answer>") == 1, f"{k} placeholder"
print("template word contract OK")

pairs = [json.loads(line) for line in open(R / "external_swap_pairs.jsonl", encoding="utf-8")]
print("retained pairs:", len(pairs))

# ---- template rendering contract ----
audit_rows = []
failed_rows = 0
for p in pairs:
    for k in ("T0", "T1", "T2"):
        co = TEMPLATES[k].replace("<answer>", p["r_o"])
        cs = TEMPLATES[k].replace("<answer>", p["r_s"])
        eq = unicodedata.normalize("NFKC", co) == unicodedata.normalize("NFKC", cs)
        audit_rows.append({"source_group_id": p["source_group_id"], "split": p["split"],
                           "template": k, "c_o": co, "c_s": cs, "c_o_nfc_eq_c_s": eq})
        failed_rows += 1 if eq else 0
if failed_rows:
    sys.exit(f"STOP: {failed_rows} rows c_o==c_s (popqa_candidate_contract_invalid)")
with open(R / "candidate_template_contract_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
    w.writeheader()
    w.writerows(audit_rows)
print("template rendering contract OK for", len(audit_rows), "rows")

# ---- mechanical outputs independent of capacity ----
by_split = Counter(p["split"] for p in pairs)
rel_count = Counter(p["relation"] for p in pairs)
rel_dist = defaultdict(Counter)
for p in pairs:
    rel_dist[p["split"]][p["relation"]] += 1
total = len(pairs)
n_rel_split = {s: len(rel_dist[s]) for s in ("train", "dev", "final_reserve")}
max_share = max(rel_count.values()) / total if total else 0

with open(R / "eligible_source_groups.jsonl", "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps({"source_group_id": p["source_group_id"],
                            "source_record_id": p["source_record_id"], "split": p["split"],
                            "relation": p["relation"], "question": p["question"],
                            "canonical_answer": p["r_o"]}) + "\n")

all_rels = sorted(set(rel_count.keys()))
with open(R / "relation_distribution_by_split.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["relation", "train", "dev", "final_reserve", "total", "share_of_total"])
    for rel in all_rels:
        w.writerow([rel, rel_dist["train"].get(rel, 0), rel_dist["dev"].get(rel, 0),
                    rel_dist["final_reserve"].get(rel, 0), rel_count[rel],
                    round(rel_count[rel] / total, 6)])

spec = {
    "experiment": "E0 PopQA Relation-Controlled Swap",
    "source": {"repo": "akariasai/PopQA", "revision": "098765c79ea10a2cb19c828324e33281b8336ec0",
               "config": "default", "split": "test", "license": "not specified",
               "files_sha256": {"test.tsv": "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b",
                                "README.md": "bb04b56bc87a3b2865cc2e2a1649ba6c766a7a44dcba5a53170fbfc72c0da9f0"},
               "download_date": "2026-08-03"},
    "schema": {"question": "question", "canonical_answer": "obj", "relation": "prop", "official_id": "id"},
    "source_group_id": "SHA256(NFKC(q)\\x00NFKC(obj)\\x00NFKC(prop)\\x00NFKC(str(id)))",
    "normalization": "NFKC + collapse whitespace; keep case & punctuation",
    "filters": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
    "split": {"order": "dict-sort sgid, then random.Random(20260816) shuffle", "ratio": "60/20/20",
              "counts": {"train": by_split["train"], "dev": by_split["dev"], "final_reserve": by_split["final_reserve"]}},
    "donor": {"same_split": True, "same_relation": True, "different_sgid": True, "answer_differs_norm": True,
              "rng": "random.Random('20260816|'+source_group_id), choice()",
              "candidates_sorted_by_sgid": True, "fixed_across_templates": True},
    "templates": {"T0": TEMPLATES["T0"], "T1": TEMPLATES["T1"], "T2": TEMPLATES["T2"],
                  "sha256": EXPECTED_SHA, "inherited_from": "D1-R-A canonical"},
    "four_cell_contract": {"OO": ["r_o", "c_o", "Accept"], "OS": ["r_o", "c_s", "Reject"],
                           "SO": ["r_s", "c_o", "Reject"], "SS": ["r_s", "c_s", "Accept"]},
    "capacity_gates": {"total_min": 1200, "split_min": {"train": 720, "dev": 240, "final_reserve": 240},
                       "relations_per_split_min": 20, "max_relation_share": 0.25,
                       "achieved": {"total": total, "train": by_split["train"], "dev": by_split["dev"],
                                    "final_reserve": by_split["final_reserve"],
                                    "relations_per_split": n_rel_split, "max_relation_share": max_share}},
    "blind_packet": {"split": "dev", "seed": 20260817, "n": 100, "produced": False,
                     "fields": ["audit_id", "question", "r_o", "r_s", "T0_c_o", "T0_c_s", "T1_c_o", "T1_c_s", "T2_c_o", "T2_c_s"]},
}
(R / "relation_controlled_swap_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
(R / "relation_controlled_swap_spec.sha256").write_text(
    hashlib.sha256((R / "relation_controlled_swap_spec.json").read_bytes()).hexdigest() + "\n", encoding="utf-8")

records = [json.loads(line) for line in open(R / "scripts" / "_records.jsonl", encoding="utf-8")]
r6_examples = [rec for rec in records
               if unicodedata.normalize("NFKC", rec["canonical_answer_nfkc"]) in rec["question_nfkc"]]
lines = ["# failure_examples.md", "",
         "## 说明", "",
         "本数据资格门无行为失败样本。此处列出 R6（答案出现在问题中）被排除组的前 5 例，"
         "以及保留组模板渲染抽样（合同审计示例，非行为结果）。", ""]
for rec in r6_examples[:5]:
    lines.append(f"- R6: `{rec['source_record_id']}` relation={rec['relation_nfkc']!r}："
                 f"answer 出现在 question 中 (a={rec['canonical_answer_nfkc'][:20]!r})")
lines += ["", "## 保留组模板渲染抽样", ""]
for p in pairs[:5]:
    lines.append(f"- `{p['source_group_id'][:8]}` [{p['split']}] {p['relation']}: "
                 f"T0 c_o={TEMPLATES['T0'].replace('<answer>', p['r_o'])!r}  "
                 f"T0 c_s={TEMPLATES['T0'].replace('<answer>', p['r_s'])!r}")
(R / "failure_examples.md").write_text("\n".join(lines), encoding="utf-8")

# ---- capacity check ----
cap_ok = (total >= 1200 and by_split["train"] >= 720 and by_split["dev"] >= 240
          and by_split["final_reserve"] >= 240
          and all(n_rel_split[s] >= 20 for s in n_rel_split)
          and max_share <= 0.25)
print("capacity:", "PASS" if cap_ok else "FAIL",
      {"total": total, "splits": dict(by_split), "n_rel_split": n_rel_split, "max_share": max_share})

(R / "artifacts").mkdir(parents=True, exist_ok=True)

if cap_ok:
    # blind packet (dev, seed 20260817, 100 groups)
    dev_pairs = [p for p in pairs if p["split"] == "dev"]
    rng = random.Random(20260817)
    sample = rng.sample(dev_pairs, 100)
    packet_rows = []
    for i, p in enumerate(sample, 1):
        packet_rows.append({"audit_id": f"P{i:03d}", "question": p["question"],
                            "r_o": p["r_o"], "r_s": p["r_s"],
                            "T0_c_o": TEMPLATES["T0"].replace("<answer>", p["r_o"]),
                            "T0_c_s": TEMPLATES["T0"].replace("<answer>", p["r_s"]),
                            "T1_c_o": TEMPLATES["T1"].replace("<answer>", p["r_o"]),
                            "T1_c_s": TEMPLATES["T1"].replace("<answer>", p["r_s"]),
                            "T2_c_o": TEMPLATES["T2"].replace("<answer>", p["r_o"]),
                            "T2_c_s": TEMPLATES["T2"].replace("<answer>", p["r_s"])})
    with open(R / "blind_candidate_contract_packet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(packet_rows[0].keys()))
        w.writeheader()
        w.writerows(packet_rows)
    label = "popqa_relation_controlled_external_data_feasible"
    cap_report = ("是（total=%d，train=%d，dev=%d，final=%d；每 split relation=%s；max share=%.4f）"
                  % (total, by_split["train"], by_split["dev"], by_split["final_reserve"],
                     str(n_rel_split), max_share))
else:
    label = "popqa_relation_swap_capacity_insufficient"
    cap_report = ("否（total=%d，train=%d，dev=%d，final=%d；每 split relation=%s；max share=%.4f）"
                  % (total, by_split["train"], by_split["dev"], by_split["final_reserve"],
                     str(n_rel_split), max_share))

fail_reason = ""
if not cap_ok:
    reasons = []
    if total < 1200:
        reasons.append(f"total {total}<1200")
    if by_split["train"] < 720:
        reasons.append(f"train {by_split['train']}<720")
    if by_split["dev"] < 240:
        reasons.append(f"dev {by_split['dev']}<240")
    if by_split["final_reserve"] < 240:
        reasons.append(f"final {by_split['final_reserve']}<240")
    if any(n_rel_split[s] < 20 for s in n_rel_split):
        reasons.append(f"relations_per_split {n_rel_split} (PopQA 官方仅 16 个 relation 类型)")
    if max_share > 0.25:
        reasons.append(f"max_share {max_share:.4f}>0.25")
    fail_reason = "; ".join(reasons)

# ---- final_report.md ----
def result_cell(ok, detail=""):
    return f"是{('（'+detail+'）') if detail else ''}" if ok else f"否{('（'+detail+'）') if detail else ''}"

(R / "final_report.md").write_text(
    f"""# E0：PopQA Relation-Controlled Swap 外部数据资格门 — 最终报告

## 结果总表

| 问题 | 结果 |
|---|---|
| PopQA 官方数据是否可唯一获取？ | 是（hf-mirror，commit 098765c7，SHA256 已记录） |
| question / answer / relation-property 契约是否有效？ | 是（question/obj/prop/id，14,267 行唯一） |
| 是否在 split 内完成 relation-controlled donor 选择？ | 是（train/dev/final-reserve 各自 split 内，同 relation，r_s≠r_o） |
| 四格与 T0/T1/T2 Candidate 合同是否全部通过？ | 是（四格共享 q/r_o/r_s/c_o/c_s；模板逐字继承且 SHA256 一致） |
| 是否达到外部确认容量门槛？ | {cap_report} |
| 是否加载任何 Judge 或运行任何推理？ | 否 |
| 是否读取旧实验 train/final-reserve 文本？ | 否 |
| 最终标签 | {label} |

## 结果

{'容量门全部通过。' if cap_ok else '容量门未通过：' + fail_reason}

## 方法

- 数据：PopQA 官方 `test` split（14,267 行），字段 question / obj / prop / id。
- source_group_id = SHA256(NFKC(q) ∥ '\\x00' ∥ NFKC(obj) ∥ '\\x00' ∥ NFKC(prop) ∥ '\\x00' ∥ NFKC(str(id)))，唯一。
- 机械过滤 R1–R6（记录级）→ R7/R8（split 内 donor 相关）；R4 排除 1 组，R6 排除 189 组。
- 先切分（dict 排序 + Random(20260816) shuffle，60/20/20），再在 split 内选 donor（同 relation、sgid 不同、答案规范化后不同；每 group 独立 RNG `20260816|sgid`，choice 一次）。
- 四格合同：OO/OS/SO/SS 各用 r_o/r_s 与 c_o/c_s，全部机械验证。
- Candidate 模板仅继承 D1-R-A canonical T0/T1/T2，逐字与 SHA256 对照，无新模板。

## 过滤漏斗

| 阶段 | 排除 | 剩余 |
|---|---|---|
| 初始 | 0 | 14,267 |
| R1 空字段 | 0 | 14,267 |
| R2 控制字符 | 0 | 14,267 |
| R3 答案长度 | 0 | 14,267 |
| R4 答案 token 数 | 1 | 14,266 |
| R5 问题 token 数 | 0 | 14,266 |
| R6 答案出现在问题中 | 189 | 14,077 |
| R7 split 内 relation<2 | 0 | 14,077 |
| R8 donor 答案相同 | 0 | 14,077 |
| 最终保留 |  | 14,077 |

## 容量

- 总保留 14,077 ≥ 1,200 ✓；train 8,446 ≥ 720 ✓；dev 2,815 ≥ 240 ✓；final-reserve 2,816 ≥ 240 ✓。
- **每 split distinct relation = 16 < 20 ✗**（PopQA 官方仅 16 个 relation/property 类型，任何 split 无法达到 ≥20）。
- 最大 relation 占比 {max_share:.4f} ≤ 0.25 ✓。

按协议，容量不足时不得放宽过滤、改变切分比例、换 seed、跨 split 选 donor、引入第二数据集或降低门槛；
因此输出 `{label}` 并停止。

## 盲审计包

{'已从 development split 以 seed 20260817 抽取 100 组 → blind_candidate_contract_packet.csv。'
 if cap_ok else '容量门未通过，不生成 blind_candidate_contract_packet.csv。'}

## 边界与局限

- PopQA 官方仅单一 split；本实验按协议 2.1 自行切分 train/dev/final-reserve。
- 本构造是 JAR-style、relation-controlled 的外部 swap，不声称复现 JAR 原始 type-preserving pipeline。
- license 未声明（not specified）。
""", encoding="utf-8")

(R / "artifacts" / "decision.json").write_text(json.dumps({
    "final_label": label, "capacity_ok": cap_ok,
    "capacity_failure_reason": fail_reason if not cap_ok else None,
    "source_ok": True, "schema_ok": True, "donor_in_split_ok": True,
    "four_cell_contract_ok": True, "template_contract_ok": True,
    "capacity": {"total": total, "train": by_split["train"], "dev": by_split["dev"],
                 "final_reserve": by_split["final_reserve"],
                 "relations_per_split": n_rel_split, "max_relation_share": max_share},
    "funnel": {"initial": 14267, "R1_R2_R3": 0, "R4": 1, "R5": 0, "R6": 189, "R7": 0, "R8": 0,
               "retained": 14077},
    "judge_loaded": False, "inference_run": False, "hidden_states_read": False,
    "probe_trained": False, "old_train_final_reserve_read": False,
    "blind_packet_produced": cap_ok}, indent=2), encoding="utf-8")

print("final_label:", label)
print("deliverables written")
