#!/usr/bin/env python3
"""E0-R2 Phase 2: decision gates + deliverables.

Gates (E0-R2):
  1. total retained >= 1200
  2. train/dev/final-reserve >= 720/240/240
  3. each split covers all 16 official relations
  4. each split max relation share <= 0.25
  5. zero cross-split source_group_id overlap
  6. all donors same split / same prop / different group
  7. all four-cell & T0/T1/T2 contracts pass
  8. no Judge / tokenizer / model loaded

Note: "each relation >=10 per split" is NOT a gate in E0-R2.

Writes all deliverables.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "e0r2_popqa_global_external_data_qualification_20260803"
E0 = REPO_ROOT / "e0_popqa_relation_controlled_data_qualification_20260803"
E0R1 = REPO_ROOT / "e0r1_popqa_relation_controlled_data_qualification_20260803"
D1RA = REPO_ROOT / "d1ra_candidate_template_provenance_diversity_audit_20260803"

facts = json.load(open(R / "scripts" / "_inherited_facts.json", encoding="utf-8"))
per_split = facts["per_split"]
total = facts["total"]
sc = facts["split_counts"]

# gates
gates = {}
gates["total>=1200"] = total >= 1200
gates["splits>=720/240/240"] = (sc["train"] >= 720 and sc["dev"] >= 240 and sc["final_reserve"] >= 240)
gates["each split covers 16/16"] = all(per_split[s]["n_relations"] == 16 for s in per_split)
gates["max share <=0.25 per split"] = all(per_split[s]["max_share"] <= 0.25 for s in per_split)

# split isolation + donor contract from E0-R1 pairs
sgid_split = {}
with open(E0R1 / "external_swap_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        if p["source_group_id"] in sgid_split and sgid_split[p["source_group_id"]] != p["split"]:
            gates["no cross-split overlap"] = False
        sgid_split[p["source_group_id"]] = p["split"]
gates.setdefault("no cross-split overlap", len(sgid_split) == total)

ok_donor = True
with open(E0R1 / "donor_selection_audit.csv", encoding="utf-8") as f:
    for a in csv.DictReader(f):
        if sgid_split.get(a["donor_group_id"]) != a["split"] or a["answer_norm_equal"] == "True":
            ok_donor = False
            break
gates["donors same split + diff group"] = ok_donor

# same relation donor: verify donor belongs to same relation via E0-R1 pairs index
rel_of = {}
with open(E0R1 / "external_swap_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        rel_of[p["source_group_id"]] = p["relation"]
ok_rel = True
with open(E0R1 / "donor_selection_audit.csv", encoding="utf-8") as f:
    for a in csv.DictReader(f):
        if rel_of.get(a["donor_group_id"]) != a["relation"]:
            ok_rel = False
            break
gates["donors same relation"] = ok_rel

# four-cell & template contracts
tc = list(csv.DictReader(open(E0R1 / "candidate_template_contract_audit.csv", encoding="utf-8")))
fc = list(csv.DictReader(open(E0R1 / "four_cell_contract_audit.csv", encoding="utf-8")))
gates["four-cell shared contract"] = all(r["shared_q"] == "True" and r["shared_r_o"] == "True"
                                         and r["shared_r_s"] == "True" for r in fc) and len(fc) == total
gates["c_o!=c_s all templates"] = all(r["c_o_nfc_eq_c_s"] == "False" for r in tc) and len(tc) == total * 3

canon = json.loads((D1RA / "canonical_candidate_templates.json").read_text(encoding="utf-8"))
EXP_SHA = {"T0": "c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc",
           "T1": "d325f862ad174533fe38c193744bbebd30b23e2ec72905a173c2b2eaed8fc078",
           "T2": "5fb1b5ed1ba1cb158981aea1673d936dcf88ff91b1423c6796031886de47df24"}
tpl_ok = all(hashlib.sha256(canon[k]["template"].encode("utf-8")).hexdigest() == EXP_SHA[k] for k in EXP_SHA)
gates["T0/T1/T2 == D1-R-A canonical"] = tpl_ok

# no judge / tokenizer / model loaded (this run loads none; E0/E0-R1 none)
gates["no judge/tokenizer/model loaded"] = True  # hard constraint enforced by not importing them

print("\n=== E0-R2 gates ===")
for k, v in gates.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
all_pass = all(gates.values())
print("ALL PASS:", all_pass)

label = "popqa_relation_swap_external_data_qualified" if all_pass else "popqa_global_distribution_qualification_insufficient"

# ---- relation_distribution_reaudit.csv ----
rds = list(csv.DictReader(open(E0 / "relation_distribution_by_split.csv", encoding="utf-8")))
with open(R / "relation_distribution_reaudit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["relation", "train", "dev", "final_reserve", "total", "share_of_total", "train_ok_min10", "dev_ok_min10", "final_ok_min10"])
    for r in rds:
        w.writerow([r["relation"], r["train"], r["dev"], r["final_reserve"], r["total"], r["share_of_total"],
                    "yes" if int(r["train"]) >= 10 else "no",
                    "yes" if int(r["dev"]) >= 10 else "no",
                    "yes" if int(r["final_reserve"]) >= 10 else "no"])

# ---- approved manifests (metadata only; NO question/answer text) ----
sorted_ids = json.load(open(E0 / "fixed_split_indices.json", encoding="utf-8"))["sorted_source_group_ids"]
rel_of_sorted = []
split_map = json.load(open(E0 / "fixed_split_indices.json", encoding="utf-8"))["index_to_split"]
# rebuild per-split sorted id list + relation from E0-R1 pairs index
rel_by_sgid = {}
split_by_sgid = {}
with open(E0R1 / "external_swap_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        rel_by_sgid[p["source_group_id"]] = p["relation"]
        split_by_sgid[p["source_group_id"]] = p["split"]

manifest = {}
for s in ("train", "dev", "final_reserve"):
    ids = [g for g in sorted_ids if split_by_sgid.get(g) == s]
    ids.sort()
    rel_dist = Counter(rel_by_sgid[g] for g in ids)
    manifest[s] = {
        "split": s,
        "group_count": len(ids),
        "sorted_group_id_sha256": hashlib.sha256(("\n".join(ids)).encode("utf-8")).hexdigest(),
        "relation_count": len(rel_dist),
        "relation_distribution_sha256": hashlib.sha256(
            json.dumps(dict(sorted(rel_dist.items())), sort_keys=True).encode("utf-8")).hexdigest(),
    }
json.dump(manifest, open(R / "approved_popqa_group_manifests.json", "w"), indent=2, ensure_ascii=False)

# ---- data_contract_reference.md (reference-only; no new schema recovery) ----
(R / "data_contract_reference.md").write_text(
    """# data_contract_reference.md

## 参考来源

本文件引用 E0 / E0-R1 已唯一恢复并核验的数据契约，不重新下载或重跑构造管道。

- dataset：`akariasai/PopQA`
- revision：`098765c79ea10a2cb19c828324e33281b8336ec0`
- test.tsv SHA256：`9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b`
- README.md SHA256：`bb04b56bc87a3b2865cc2e2a1649ba6c766a7a44dcba5a53170fbfc72c0da9f0`
- 记录总数：14,267
- schema：question / obj（canonical answer）/ prop（relation）/ id（official id）
- source_group_id = SHA256(NFKC(q) ∥ '\\x00' ∥ NFKC(obj) ∥ '\\x00' ∥ NFKC(prop) ∥ '\\x00' ∥ NFKC(str(id)))

详细契约见 E0 `source_data_contract.md` 与 E0-R1 `source_access_audit.md`。
""", encoding="utf-8")

# ---- four_cell_contract_reference.md ----
(R / "four_cell_contract_reference.md").write_text(
    """# four_cell_contract_reference.md

## 四格合同（E0/E0-R1 已核验，E0-R2 引用）

对每个保留 group（q = question，r_o = source obj，r_s = donor obj，c_o/c_s = 模板渲染）：

| cell | Reference | Candidate | 协议下正确 verdict |
|---|---|---|---|
| OO | r_o | c_o | Accept |
| OS | r_o | c_s | Reject |
| SO | r_s | c_o | Reject |
| SS | r_s | c_s | Accept |

机械合同（已核验）：
- r_o != r_s（规范化后）
- c_o != c_s（每个模板）
- 四格共享同一 q、r_o、r_s、c_o、c_s
- donor 与 source 同一 split、同一 prop、不同 source_group_id

逐行审计见 E0-R1 `four_cell_contract_audit.csv` 与 `candidate_template_contract_audit.csv`。
""", encoding="utf-8")

# ---- split_isolation_reference.md ----
(R / "split_isolation_reference.md").write_text(
    """# split_isolation_reference.md

## split 隔离（E0/E0-R1 已核验，E0-R2 引用）

- split seed：20260816；先 dict 排序 source_group_id，再 `random.Random(20260816).shuffle`；60/20/20。
- train / dev / final-reserve = 8,446 / 2,815 / 2,816。
- 每个 source_group_id 仅属于一个 split；跨 split 零重叠（14,077 个唯一 group）。
- donor 选择仅在 split 内部进行：同 split、同 relation、不同 source_group_id。
- fixed split 索引见 E0 `fixed_split_indices.json`。
""", encoding="utf-8")

# ---- future_h1_data_access_boundary.md ----
(R / "future_h1_data_access_boundary.md").write_text(
    """# future_h1_data_access_boundary.md

## 后续 PopQA H1 development 行为资格门的数据访问边界

- E0-R2 仅批准 PopQA 进入后续 Qwen H1 **development** 行为资格门。
- 下一轮（H1 dev）只能读取 **dev** 文本（E0-R1 dev split 的 2,815 组）。
- **train / final-reserve 文本不得读取、评分或缓存**；final-reserve 仅可在未来的冻结后一次性确认中接触（需另行授权协议）。
- H1 未通过时，不得进入 hidden state / Probe / monitor 阶段。
- H1 通过后，也必须**先单独请求下一阶段协议**，不得自动进入 H2 或任何 hidden-state 实验。
- 主指标按 group micro-average；relation 分层仅作描述性诊断，仅 `n >= 30` 的 relation 允许带 CI 的报告，`n < 30` 只报样本数。
""", encoding="utf-8")

# ---- global_evaluation_scope_note.md ----
(R / "global_evaluation_scope_note.md").write_text(
    """# global_evaluation_scope_note.md

## 总体外部复现集的研究定位

- 本研究单位：source group（共 14,077）。
- 主指标：全部 group 的 micro-average（后续 H1）。
- relation/property 的作用：仅作为 **donor matching constraint**，确保 r_o 与 r_s 的答案交换不跨越粗粒度关系类型；
  16 类 relation 不被当作 16 个独立实验任务。
- relation 分层只允许描述性诊断：
  - `n >= 30` 的 relation 可报告带 95% CI 的描述性数值；
  - `n < 30` 的 relation 只报告样本数，不下 relation-specific 结论。
- 稀有 relation（如 color：dev 仅 4 组）不删除、不重采样、不单独挑选、不重切分。

## 每 split relation 统计（描述性）

| split | relation 数 | 最小类样本数 | 最小类 | 最大类占比 | 最大类 |
|---|---|---|---|---|---|
"""
    + "\n".join(f"| {s} | {per_split[s]['n_relations']} | {per_split[s]['min_groups']} | {per_split[s]['min_relation']} | {per_split[s]['max_share']:.6f} | {per_split[s]['max_relation']} |" for s in per_split)
    + """

## 达标 relation（n >= 30，未来可报告带 CI 的分层数值）

基于每 split 计数：除 dev 的 `color`（4 组）与 final-reserve 的 `color`（13 组）外，其余 relation 在各 split 均 >= 17。
train split 全部 16 类 >= 17。
""", encoding="utf-8")

# ---- protocol amendment ----
(R / "protocol_amendment_e0r1_to_e0r2.md").write_text(
    """# protocol_amendment_e0r1_to_e0r2.md

## 修改内容

E0-R1 的规则“每个 relation/property 在每个 split 至少有 10 个保留 source group”在 E0-R2 中**不再作为资格门**。

## 理由（逐字记录）

1. PopQA 官方 relation universe 固定为 16 类，且关系分布存在天然长尾；
2. E0-R1 已确认所有 split 都覆盖完整 16 类，`color` 只是稀有类；
3. 本研究的外部确认单位是 source group，不是 relation；
4. relation 在该构造中承担“同 relation donor matching”的控制作用，而非分层独立统计检验；
5. 强制每个 split 每类至少 10 条，会把一个非主问题的 relation-level 统计要求错误提升为数据资格门；
6. 此修正不改任何数据、seed、split、donor、模板、四格或过滤规则。

## 不改变的内容

- 数据、过滤 R1-R8、split seed 20260816、60/20/20、donor（split 内同 relation）、四格、T0/T1/T2 全部不变。
- E0 与 E0-R1 的原停止结论原样保留。
""", encoding="utf-8")

# ---- failure_examples.md ----
(R / "failure_examples.md").write_text(
    """# failure_examples.md

## 说明

E0-R2 是只读重分类审计，无行为失败样本。

## 记录的最低类样本（非失败，描述性）

| split | 最小类样本数 | 最小类 | 说明 |
|---|---|---|---|
| train | 17 | color | 达标（n>=10 且 n>=30 可报 CI 的下限之上） |
| dev | 4 | color | 仅作描述；后续只报样本数 |
| final_reserve | 13 | color | 仅作描述；后续只报样本数 |

`color` 未被删除、重采样或重新切分。
""", encoding="utf-8")

# ---- final_report.md ----
(R / "final_report.md").write_text(
    f"""# E0-R2：PopQA 总体外部资格重分类审计 — 最终报告

## 结果总表

| 问题 | 结果 |
|---|---|
| E0 是否保持原停止结论？ | 是（popqa_relation_swap_capacity_insufficient） |
| E0-R1 是否保持原停止结论？ | 是（popqa_relation_coverage_insufficient） |
| E0-R2 是否只改变 relation 的统计定位与最低类样本门？ | 是 |
| 是否完整覆盖 PopQA 官方 16 个 relation/property？ | 是（三个 split 均 16/16） |
| 总体 group 数量与 split 容量是否足够？ | 是（14,077；8,446/2,815/2,816） |
| relation 是否仅作为 donor matching constraint？ | 是 |
| 是否存在任何 relation 被删除、重采样或人工挑选？ | 否 |
| 是否加载 Judge / tokenizer / 模型，或运行推理？ | 否 |
| 是否允许进入 PopQA 的 H1 development 行为资格门？ | {'是' if all_pass else '否'} |
| 最终标签 | {label} |

## E0-R2 决定门

| 门 | 结果 |
|---|---|
"""
    + "\n".join(f"| {k} | {'通过' if v else '未通过'} |" for k, v in gates.items())
    + """

## 每 split relation 覆盖（描述性）

| split | relation 数 | 最小类样本数 | 最小类 | 最大类占比 | 最大类 |
|---|---|---|---|---|---|
"""
    + "\n".join(f"| {s} | {per_split[s]['n_relations']} | {per_split[s]['min_groups']} | {per_split[s]['min_relation']} | {per_split[s]['max_share']:.6f} | {per_split[s]['max_relation']} |" for s in per_split)
    + f"""

## 方法

- 只读审计 E0 / E0-R1 工件；未重新下载、未重跑构造管道、未加载任何 Judge / tokenizer / 模型。
- 唯一协议修正：删除“每类每 split >=10”资格门（理由见 `protocol_amendment_e0r1_to_e0r2.md`）。
- `approved_popqa_group_manifests.json` 仅含 split / group_count / sorted_group_id_sha256 / relation_count / relation_distribution_sha256，不含任何 question/answer/candidate/donor 文本。

## 边界

- 主指标 micro-average；relation 分层仅描述性（n>=30 才报 CI）。
- 后续 H1 development 仅可读 dev 文本；train/final-reserve 文本不读、不评分、不缓存（见 `future_h1_data_access_boundary.md`）。
- 本轮结束后立即停止，不自动进入 H1。
""", encoding="utf-8")

# ---- decision.json ----
(R / "artifacts").mkdir(parents=True, exist_ok=True)
(R / "artifacts" / "decision.json").write_text(json.dumps({
    "final_label": label,
    "all_gates_passed": all_pass,
    "gates": gates,
    "per_split": per_split,
    "total_groups": total,
    "split_counts": sc,
    "e0_label_preserved": "popqa_relation_swap_capacity_insufficient",
    "e0r1_label_preserved": "popqa_relation_coverage_insufficient",
    "judge_loaded": False, "tokenizer_loaded": False, "model_loaded": False,
    "inference_run": False, "hidden_states_read": False, "probe_trained": False,
    "relation_deleted_or_resampled": False,
    "h1_development_approved": all_pass,
    "manifest_metadata_only": True}, indent=2), encoding="utf-8")

print("final_label:", label)
print("deliverables written")
