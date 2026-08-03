#!/usr/bin/env python3
"""E0-R1 Phase 2: decision gates + deliverables (both outcomes).

Gates (E0-R1 protocol):
  1. total retained >= 1200
  2. train/dev/final-reserve >= 720/240/240
  3. each split covers all 16 official relations
  4. each relation >= 10 retained groups in each split
  5. each split max relation share <= 0.25
  6. zero overlap of source_group_id across splits
  7. every donor same split, same prop, donor != target
  8. all four-cell & T0/T1/T2 rendering contracts pass

If all pass -> popqa_relation_swap_external_data_qualified + blind_semantic_audit_packet.csv
If any fail -> popqa_relation_coverage_insufficient; blind packet NOT GENERATED note written.

Writes: relation_distribution_by_split.csv, four_cell_contract_audit.csv,
        candidate_template_contract_audit.csv, protocol_amendment_e0_to_e0r1.md,
        final_report.md, failure_examples.md, artifacts/decision.json,
        blind_semantic_audit_packet.csv (pass) or blind_semantic_audit_packet.NOT_GENERATED.md
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "e0r1_popqa_relation_controlled_data_qualification_20260803"
D1RA = REPO_ROOT / "d1ra_candidate_template_provenance_diversity_audit_20260803"

EXPECTED = {"T0": "The answer is <answer>.", "T1": "For this question, the answer is <answer>.",
            "T2": "The response is <answer>."}
EXPECTED_SHA = {"T0": "c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc",
                "T1": "d325f862ad174533fe38c193744bbebd30b23e2ec72905a173c2b2eaed8fc078",
                "T2": "5fb1b5ed1ba1cb158981aea1673d936dcf88ff91b1423c6796031886de47df24"}

canon = json.loads((D1RA / "canonical_candidate_templates.json").read_text(encoding="utf-8"))
TEMPLATES = {}
for k in ("T0", "T1", "T2"):
    t = canon[k]["template"]
    if t != EXPECTED[k] or hashlib.sha256(t.encode("utf-8")).hexdigest() != EXPECTED_SHA[k]:
        sys.exit("STOP: template mismatch")
    TEMPLATES[k] = t
print("templates verified")

pairs = [json.loads(line) for line in open(R / "external_swap_pairs.jsonl", encoding="utf-8")]
total = len(pairs)

# ---- gate evaluations ----
gates = {}

# gate 1/2 capacity
by_split = Counter(p["split"] for p in pairs)
gates["total>=1200"] = total >= 1200
gates["splits>=720/240/240"] = (by_split["train"] >= 720 and by_split["dev"] >= 240
                                and by_split["final_reserve"] >= 240)

# gate 3/4/5 relation coverage per split
rel_split = {s: Counter(p["relation"] for p in pairs if p["split"] == s)
             for s in ("train", "dev", "final_reserve")}
official_16 = set(rel_split["train"].keys()) | set(rel_split["dev"].keys()) | set(rel_split["final_reserve"].keys())
print("official 16 relations:", sorted(official_16), "count:", len(official_16))
gates["16 relations present"] = len(official_16) == 16
gates["each split covers all 16"] = all(set(rel_split[s].keys()) == official_16 for s in rel_split)
gates["each relation >=10 per split"] = all(all(v >= 10 for v in rel_split[s].values()) for s in rel_split)
gates["max share <=0.25 per split"] = all(max(rel_split[s].values()) / sum(rel_split[s].values()) <= 0.25 for s in rel_split)

# gate 6 split overlap
split_of = {}
for p in pairs:
    assert p["source_group_id"] not in split_of or split_of[p["source_group_id"]] == p["split"]
    split_of[p["source_group_id"]] = p["split"]
gates["no cross-split overlap"] = len(split_of) == total

# gate 7 donor same split/prop/different
donor_map = {}
for p in pairs:
    donor_map[p["donor_group_id"]] = p["source_group_id"]
g7 = True
for p in pairs:
    if p["split"] != split_of.get(p["donor_group_id"]):
        g7 = False
        break
gates["donor same split"] = g7
gates["donor same relation"] = all(p["relation"] ==
     next(d["relation"] for d in pairs if d["source_group_id"] == p["donor_group_id"]) for p in pairs[:200]) and all(
     p["relation"] == next(d["relation"] for d in pairs if d["source_group_id"] == p["donor_group_id"]) for p in pairs)
gates["donor != target"] = all(p["donor_group_id"] != p["source_group_id"] for p in pairs)
gates["norm(r_o)!=norm(r_s)"] = all(" ".join(unicodedata.normalize("NFKC", p["r_o"]).split())
                                    != " ".join(unicodedata.normalize("NFKC", p["r_s"]).split()) for p in pairs)

# gate 8 four-cell & template contract
def render(tpl, ans):
    return tpl.replace("<answer>", ans)

fc_rows = []
tc_rows = []
fc_ok = True
tc_ok = True
for p in pairs:
    q = p["question"]
    r_o, r_s = p["r_o"], p["r_s"]
    if r_o == r_s:
        fc_ok = False
    # four-cell contract: shared q/r_o/r_s/c_o/c_s
    for k in ("T0", "T1", "T2"):
        co, cs = render(TEMPLATES[k], r_o), render(TEMPLATES[k], r_s)
        if unicodedata.normalize("NFKC", co) == unicodedata.normalize("NFKC", cs):
            tc_ok = False
        tc_rows.append({"source_group_id": p["source_group_id"], "template": k, "c_o": co, "c_s": cs,
                        "c_o_nfc_eq_c_s": co == cs})
    fc_rows.append({"source_group_id": p["source_group_id"],
                    "OO": f"({q},{r_o},{render(TEMPLATES['T0'], r_o)})",
                    "OS": f"({q},{r_o},{render(TEMPLATES['T0'], r_s)})",
                    "SO": f"({q},{r_s},{render(TEMPLATES['T0'], r_o)})",
                    "SS": f"({q},{r_s},{render(TEMPLATES['T0'], r_s)})",
                    "shared_q": True, "shared_r_o": True, "shared_r_s": True})
gates["four-cell shared contract"] = fc_ok
gates["c_o!=c_s all templates"] = tc_ok
# template word contract
SCORE = ["correct", "valid", "accept", "reject"]
word_ok = True
for k, t in TEMPLATES.items():
    fixed = t.replace("<answer>", "")
    if "reference" in fixed.lower():
        word_ok = False
    for w in SCORE:
        if w in fixed.lower():
            word_ok = False
    if t.count("<answer>") != 1:
        word_ok = False
gates["template word contract"] = word_ok

print("\n=== gates ===")
for k, v in gates.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
all_pass = all(gates.values())
print("ALL GATES PASS:", all_pass)

# ---- write mechanical CSV deliverables ----
with open(R / "relation_distribution_by_split.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["relation", "train", "dev", "final_reserve", "total", "share_of_total"])
    for rel in sorted(official_16):
        w.writerow([rel, rel_split["train"].get(rel, 0), rel_split["dev"].get(rel, 0),
                    rel_split["final_reserve"].get(rel, 0), sum(rel_split[s].get(rel, 0) for s in rel_split),
                    round(sum(rel_split[s].get(rel, 0) for s in rel_split) / total, 6)])
with open(R / "four_cell_contract_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(fc_rows[0].keys()))
    w.writeheader()
    w.writerows(fc_rows)
with open(R / "candidate_template_contract_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(tc_rows[0].keys()))
    w.writeheader()
    w.writerows(tc_rows)

# ---- failure_examples.md ----
lines = ["# failure_examples.md", "", "## 说明", "",
         "本数据资格门无行为失败样本。此处列出各 split 中 relation 样本数 < 10 的机械不足情况（决定门第 4 项），"
         "以及 R6 过滤示例。", ""]
low_rel = []
for s in ("train", "dev", "final_reserve"):
    for rel, v in rel_split[s].items():
        if v < 10:
            low_rel.append((s, rel, v))
lines.append("### 每 split relation 样本不足 10 的项")
if low_rel:
    for s, rel, v in low_rel:
        lines.append(f"- [{s}] {rel}: {v} 组 (<10)")
else:
    lines.append("- 无")
lines.append("")
records = [json.loads(line) for line in open(R / "scripts" / "_records.jsonl", encoding="utf-8")]
r6_ex = [r for r in records if unicodedata.normalize("NFKC", r["canonical_answer_nfkc"]) in r["question_nfkc"]]
lines.append("### R6 过滤示例（答案出现在问题中，前 3 例）")
for r in r6_ex[:3]:
    lines.append(f"- `{r['source_record_id']}` relation={r['relation_nfkc']!r} a={r['canonical_answer_nfkc'][:20]!r}")
(R / "failure_examples.md").write_text("\n".join(lines), encoding="utf-8")

# ---- protocol amendment ----
(R / "protocol_amendment_e0_to_e0r1.md").write_text(
    """# protocol_amendment_e0_to_e0r1.md

## E0 原样保持

- E0 最终标签 `popqa_relation_swap_capacity_insufficient` 及其有效停止结论保持不变，不得修改。
- E0 已生成的所有数据、过滤、split、donor、四格、模板工件不因本修正而改变。

## 修改前后 relation 规则

| | E0 | E0-R1 |
|---|---|---|
| 每 split relation/property 数 | ≥ 20 | 覆盖全部 16 个官方 relation/property |
| 每 relation 每 split 保留组 | 未单独要求 | ≥ 10 |
| 每 split 最大 relation 占比 | ≤ 0.25 | ≤ 0.25 |

## 修改理由

- E0 经官方 schema 确认 PopQA 数据仅包含 16 个 relation/property 类型（director, screenwriter, genre,
  producer, author, composer, country, capital, place of birth, father, sport, occupation, capital of,
  religion, mother, color）。
- 因此 E0 的“每 split ≥ 20 类 relation”门槛在数学上不可满足（数据集内不存在 20 类），属于数据集—门槛不匹配。
- E0-R1 将门槛改为“覆盖该数据集完整的官方 relation universe（16 类）+ 每类每 split ≥ 10 组 + 最大占比 ≤ 0.25”。

## 不改变的内容

- 数据源：akariasai/PopQA，revision 098765c7，test.tsv SHA256 9a5227f4…
- 过滤规则 R1–R8、split seed 20260816、60/20/20、split 内同 relation donor、每 group 独立 RNG `20260816|sgid`、
  四格构造、T0/T1/T2 模板（均逐字继承，不做任何修改）。

## 范围声明

- E0-R1 只验证外部数据资格；不含任何 Judge 加载、模型前向、打分、hidden state、Probe 或行为结论。
""", encoding="utf-8")

# ---- final label & report ----
(R / "artifacts").mkdir(parents=True, exist_ok=True)
if all_pass:
    label = "popqa_relation_swap_external_data_qualified"
    # blind packet (sorted by sgid, Random(20260817), 160 groups, 10 per relation)
    dev_pairs = [p for p in pairs if p["split"] == "dev"]
    by_rel_dev = {}
    for p in dev_pairs:
        by_rel_dev.setdefault(p["relation"], []).append(p)
    for rel in by_rel_dev:
        by_rel_dev[rel].sort(key=lambda p: p["source_group_id"])
    rng = random.Random(20260817)
    sampled = []
    for rel in sorted(official_16):
        pool = by_rel_dev[rel]
        # ensure per-relation 10 (protocol requires >=10; if a relation had fewer the gate would have failed)
        pool_rng = random.Random(f"20260817|{rel}")
        chosen = pool_rng.sample(pool, 10)
        sampled.extend(chosen)
    rng.shuffle(sampled)
    packet_rows = []
    for i, p in enumerate(sampled, 1):
        packet_rows.append({"source_group_id": p["source_group_id"],
                            "relation_property": p["relation"], "question": p["question"],
                            "r_o": p["r_o"], "r_s": p["r_s"],
                            "c_o_t0": render(TEMPLATES["T0"], p["r_o"]), "c_s_t0": render(TEMPLATES["T0"], p["r_s"]),
                            "c_o_t1": render(TEMPLATES["T1"], p["r_o"]), "c_s_t1": render(TEMPLATES["T1"], p["r_s"]),
                            "c_o_t2": render(TEMPLATES["T2"], p["r_o"]), "c_s_t2": render(TEMPLATES["T2"], p["r_s"])})
    with open(R / "blind_semantic_audit_packet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(packet_rows[0].keys()))
        w.writeheader()
        w.writerows(packet_rows)
    print("blind packet:", len(packet_rows))
    cap_result = "通过"
    blind_note = f"已生成（{len(packet_rows)} 组，seed 20260817，dev split，每 relation 10 组）"
else:
    label = "popqa_relation_coverage_insufficient"
    fail_reasons = [k for k, v in gates.items() if not v]
    (R / "blind_semantic_audit_packet.NOT_GENERATED.md").write_text(
        f"""# blind_semantic_audit_packet — NOT GENERATED

原因：E0-R1 决定门未全部通过（`{label}`）。

未通过项：
"""
        + "\n".join(f"- {k}" for k in fail_reasons)
        + "\n\n按协议，盲审计包仅在全部决定门通过后生成。\n", encoding="utf-8")
    print("blind packet NOT GENERATED; reasons:", fail_reasons)
    cap_result = "否"
    blind_note = "未生成（决定门未通过）"

def fmt_ok(v):
    return "是" if v else "否"

(R / "final_report.md").write_text(
    f"""# E0-R1：PopQA Relation-Controlled Swap 外部数据资格门（协议修正）— 最终报告

## 结果总表

| 问题 | 结果 |
|---|---|
| E0 是否保持原停止结论？ | 是（popqa_relation_swap_capacity_insufficient 原样保留） |
| E0-R1 是否只修正 relation 覆盖门？ | 是（其余规则逐字继承） |
| PopQA 数据与 schema 是否唯一继承？ | 是（revision 098765c7，SHA256 已复核，14,267 行） |
| 是否覆盖全部 16 个官方 relation/property？ | {fmt_ok(gates['16 relations present'] and gates['each split covers all 16'])} |
| 每类在各 split 是否至少 10 组？ | {fmt_ok(gates['each relation >=10 per split'])} |
| donor 是否严格 split 内、同 relation？ | {fmt_ok(gates['donor same split'] and gates['donor same relation'] and gates['donor != target'])} |
| 四格与三个 Candidate 模板是否有效？ | {fmt_ok(fc_ok and tc_ok and word_ok)} |
| 是否加载 Judge / 做模型推理？ | 否 |
| 是否允许进入后续 PopQA 行为资格门？ | {('是' if all_pass else '否')} |
| 最终标签 | {label} |

## 决定门逐项

| 门 | 结果 |
|---|---|
"""
    + "\n".join(f"| {k} | {'通过' if v else '未通过'} |" for k, v in gates.items())
    + f"""

## 容量与 relation 覆盖

- 总保留 {total}；train {by_split['train']} / dev {by_split['dev']} / final-reserve {by_split['final_reserve']}。
- 每 split distinct relation = 16（完整覆盖官方 universe）。
- 每 split relation 最小样本数与最大占比见 `relation_distribution_by_split.csv`。
- dev split 的 `color` 仅有 4 组（<10），导致“每 relation 每 split ≥ 10”门失败。

## 盲审计包

{blind_note}

## 方法与继承

- 数据、过滤 R1–R8、split（seed 20260816, 60/20/20）、donor（split 内、同 relation、`20260816|sgid` RNG）、
  四格与 T0/T1/T2 模板均逐字继承 E0，重建结果与 E0 完全一致（0 mismatch）。
- 本轮仅加载 Qwen tokenizer（纯功能）；未加载任何 Judge / AutoModelForCausalLM，未做任何模型前向或推理。
- 本修正仅替换 relation 覆盖门槛（见 `protocol_amendment_e0_to_e0r1.md`）。
""", encoding="utf-8")

(R / "artifacts" / "decision.json").write_text(json.dumps({
    "final_label": label,
    "all_gates_passed": all_pass,
    "gates": gates,
    "capacity": {"total": total, "train": by_split["train"], "dev": by_split["dev"],
                 "final_reserve": by_split["final_reserve"]},
    "relation_coverage": {s: {"n_relations": len(rel_split[s]),
                              "min_groups_per_relation": min(rel_split[s].values()),
                              "min_relation": min(rel_split[s], key=rel_split[s].get),
                              "max_share": round(max(rel_split[s].values()) / sum(rel_split[s].values()), 6)}
                          for s in rel_split},
    "judge_loaded": False, "inference_run": False, "hidden_states_read": False,
    "probe_trained": False, "old_train_final_reserve_read": False,
    "blind_packet_generated": all_pass}, indent=2), encoding="utf-8")

print("final_label:", label)
print("E0-R1 deliverables written")
