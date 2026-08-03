# E0-R1：PopQA Relation-Controlled Swap 外部数据资格门（协议修正）— 最终报告

## 结果总表

| 问题 | 结果 |
|---|---|
| E0 是否保持原停止结论？ | 是（popqa_relation_swap_capacity_insufficient 原样保留） |
| E0-R1 是否只修正 relation 覆盖门？ | 是（其余规则逐字继承） |
| PopQA 数据与 schema 是否唯一继承？ | 是（revision 098765c7，SHA256 已复核，14,267 行） |
| 是否覆盖全部 16 个官方 relation/property？ | 是 |
| 每类在各 split 是否至少 10 组？ | 否 |
| donor 是否严格 split 内、同 relation？ | 是 |
| 四格与三个 Candidate 模板是否有效？ | 是 |
| 是否加载 Judge / 做模型推理？ | 否 |
| 是否允许进入后续 PopQA 行为资格门？ | 否 |
| 最终标签 | popqa_relation_coverage_insufficient |

## 决定门逐项

| 门 | 结果 |
|---|---|
| total>=1200 | 通过 |
| splits>=720/240/240 | 通过 |
| 16 relations present | 通过 |
| each split covers all 16 | 通过 |
| each relation >=10 per split | 未通过 |
| max share <=0.25 per split | 通过 |
| no cross-split overlap | 通过 |
| donor same split | 通过 |
| donor same relation | 通过 |
| donor != target | 通过 |
| norm(r_o)!=norm(r_s) | 通过 |
| four-cell shared contract | 通过 |
| c_o!=c_s all templates | 通过 |
| template word contract | 通过 |

## 容量与 relation 覆盖

- 总保留 14077；train 8446 / dev 2815 / final-reserve 2816。
- 每 split distinct relation = 16（完整覆盖官方 universe）。
- 每 split relation 最小样本数与最大占比见 `relation_distribution_by_split.csv`。
- dev split 的 `color` 仅有 4 组（<10），导致“每 relation 每 split ≥ 10”门失败。

## 盲审计包

未生成（决定门未通过）

## 方法与继承

- 数据、过滤 R1–R8、split（seed 20260816, 60/20/20）、donor（split 内、同 relation、`20260816|sgid` RNG）、
  四格与 T0/T1/T2 模板均逐字继承 E0，重建结果与 E0 完全一致（0 mismatch）。
- 本轮仅加载 Qwen tokenizer（纯功能）；未加载任何 Judge / AutoModelForCausalLM，未做任何模型前向或推理。
- 本修正仅替换 relation 覆盖门槛（见 `protocol_amendment_e0_to_e0r1.md`）。
