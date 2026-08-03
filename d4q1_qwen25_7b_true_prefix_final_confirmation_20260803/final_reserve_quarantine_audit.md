# final_reserve_quarantine_audit.md

## 隔离来源

- 已完成的 final-reserve 泄露隔离审计：D4-M0 `leak_isolation_audit.md`（2026-08-03）。
  - 该审计确认唯一 final_reserve 泄露 group：`0075758e`（在 Phase 0 探索 `head -3` 时显示）。
  - 另有两个 train 组泄露（`004c1d1f`、`015c326e`），不属于 final-reserve split，不影响本轮 final 隔离。
- 本轮**不**重新浏览 D0 raw 文件推断泄露 group；泄露 group 完全由已完成的隔离工件唯一给定。

## 隔离契约

| 项 | 值 |
|---|---|
| D0 原 final-reserve group 数 | 197 |
| 泄露 final group 数 | 1 |
| 允许使用 final group 数 | 196 |
| 泄露 group 是否在允许集合中 | 否（已剔除） |
| 允许集合索引哈希（SHA256 of sorted ids） | `e1c36f658bc4af5134961901145bb52cbf9de344dcbc127668dbf6df45711f5d` |
| D0 final split 索引哈希 | `9fe440d6cb383c5c1d7d546af4a34381342d26b5bd9424bd623528a8063fc831` |

## 允许集合与 D0 final split 的一致性

- 允许集合 = D0 `fixed_split_indices.json` 的 final_reserve 197 个 id，减去泄露的 1 个 id。
- 证明：`set(allowed) == set(D0 final_reserve) - {LEAKED}`，且排序后索引哈希一致（`e1c36f658bc4af5134961901145bb52cbf9de344dcbc127668dbf6df45711f5d`）。

## 使用约束

- 本轮只流式读取 `allowed_final_group_manifest.json` 中的 196 个 group。
- 禁止打开、评分、缓存或写出泄露 group（`0075758e`）的任何文本、id、评分或 hidden state。
- 本轮所有输出（manifest/CSV/report）只允许包含 group id / hash / 数值；不得包含 final 组题目或答案正文。
