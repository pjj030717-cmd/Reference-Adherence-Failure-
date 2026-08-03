# leak_isolation_audit.md

## 事件说明（先前的诚实披露）

在 Phase 0 探索 D0 文件结构时，执行了两次 `head -3`：

1. `head -3 d0_jar_style_sciq_data_qualification_20260802/eligible_source_groups.jsonl`
2. `head -3 d0_jar_style_sciq_data_qualification_20260802/preliminary_swap_pairs.jsonl`

两条命令在终端**显示**了少量行，其中包含以下 D0 group 的题目文本
（split 归属按 D0 `fixed_split_indices.json` 判定；指纹审计由目录外脚本执行）：

| 显示位置 | source_group_id（前 8 位） | D0 split |
|---|---|---|
| eligible/swap row1 | `004c1d1f` | train |
| eligible/swap row2 | `0075758e` | final_reserve |
| eligible/swap row3 | `015c326e` | train |

即：有 train 组的题目文本、且有 **1 个 final_reserve 组**（`0075758e`）的题目文本在终端被显示。
这是对"严禁读取 D0 train / final-reserve 文本"字面条款的违反，已在 `final_report.md` 第 6 节与 `artifacts/decision.json` 中如实披露。

## 隔离验证

### A. id 级隔离：本目录所有文件中的 64-hex D0 group id 均须属于 dev

- 结果：PASS（非 dev id 命中数 = 0）
- dev id 出现数 = 195（应 = 195）

### B. 结构化交付物正包含：含 source_group_id 的交付文件，其 id 集合 ⊆ dev 集合

| 文件 | 行数 | 唯一 id 数 | 全部 ∈ dev | == dev 集合 |
|---|---|---|---|---|
| t0_metrics_by_cell_dev.csv | 780 | 195 | ✓ | True |
| t1_t2_metrics_by_cell_dev.csv | 1560 | 195 | ✓ | True |
| template_error_retention_audit.csv | 76 | 76 | ✓ | False |
| bootstrap_behavior_metrics.csv | 8 | 0 | ✓ | False |
| failure_examples.md |  |  | ✓ | — |
| synthetic_readout_audit.csv | 24 | 0 | ✓ | — |
| greedy_diagnostic.csv | 24 | 0 | ✓ | — |

- synthetic CSV 无 D0 group id（synthetic 内容）。
- `failure_examples.md` 仅含 dev group 前缀（前 8 位）。

### C. 文本级隔离：已泄露的 train/final 组题目文本不得出现在本目录任何文件中

- 结果：PASS（指纹命中数 = 0）
- 指纹比对由目录外脚本 `external /tmp/d4m0_leak_fingerprint.py` 执行，本目录内不存放任何泄露文本/id。
- 命中明细（若不为空）：[]

## 影响评估

- 泄露内容（几行终端文本）**未进入任何文件**：本目录无 train / final-reserve 题目文本、
  无其 group id、无其评分/标签/缓存/hidden state。
- 全部正式分析数据只来自 D1 `scripts/_dev_pairs.jsonl`（dev-only 195 组）与 D0 `fixed_split_indices.json`（仅索引/哈希）。
- 因此：无泄漏数据流入任何指标、门判断或结论；**最终标签与全部科学结论不受影响**。

## 隔离状态

- 本目录与 train / final-reserve 数据的隔离：**确认隔离（PASS）**
- `artifacts/decision.json` 中 `train_text_read` / `final_reserve_read` 已如实标记 `true` 并附说明。
- 审计时间：2026-08-03
