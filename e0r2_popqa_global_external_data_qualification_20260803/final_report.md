# E0-R2：PopQA 总体外部资格重分类审计 — 最终报告

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
| 是否允许进入 PopQA 的 H1 development 行为资格门？ | 是 |
| 最终标签 | popqa_relation_swap_external_data_qualified |

## E0-R2 决定门

| 门 | 结果 |
|---|---|
| total>=1200 | 通过 |
| splits>=720/240/240 | 通过 |
| each split covers 16/16 | 通过 |
| max share <=0.25 per split | 通过 |
| no cross-split overlap | 通过 |
| donors same split + diff group | 通过 |
| donors same relation | 通过 |
| four-cell shared contract | 通过 |
| c_o!=c_s all templates | 通过 |
| T0/T1/T2 == D1-R-A canonical | 通过 |
| no judge/tokenizer/model loaded | 通过 |

## 每 split relation 覆盖（描述性）

| split | relation 数 | 最小类样本数 | 最小类 | 最大类占比 | 最大类 |
|---|---|---|---|---|---|
| train | 16 | 17 | color | 0.142908 | screenwriter |
| dev | 16 | 4 | color | 0.148845 | director |
| final_reserve | 16 | 13 | color | 0.142401 | screenwriter |

## 方法

- 只读审计 E0 / E0-R1 工件；未重新下载、未重跑构造管道、未加载任何 Judge / tokenizer / 模型。
- 唯一协议修正：删除“每类每 split >=10”资格门（理由见 `protocol_amendment_e0r1_to_e0r2.md`）。
- `approved_popqa_group_manifests.json` 仅含 split / group_count / sorted_group_id_sha256 / relation_count / relation_distribution_sha256，不含任何 question/answer/candidate/donor 文本。

## 边界

- 主指标 micro-average；relation 分层仅描述性（n>=30 才报 CI）。
- 后续 H1 development 仅可读 dev 文本；train/final-reserve 文本不读、不评分、不缓存（见 `future_h1_data_access_boundary.md`）。
- 本轮结束后立即停止，不自动进入 H1。
