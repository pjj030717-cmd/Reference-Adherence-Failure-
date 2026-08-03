# final_reserve_zero_access_audit.md

- D0 split：train=587 / dev=195 / final_reserve=197。
- 本轮唯一合法 group 集合为 train_union_dev（782 个），源自 D2-R1 标签文件并校验 split sha 与 D0 记录一致。
- 本目录所有文件（.py/.csv/.json/.md/.log/.sha256）中出现的 64-hex group id 均已核验 ⊆ train_union_dev：
  `foreign gid count = 0`。
- final-reserve group id 出现次数：**0**。
- D0 swap pairs 读取时仅流式保留 split∈{train, dev} 行，final_reserve 行立即丢弃，未打印/保存/统计。
- D0 `fixed_split_indices.json` 仅以正则提取 `split_sha256`（train/dev），未加载 `groups.final_reserve` 列表。
- D4-Q1 `prefix_hidden_states/final_*.npz`（196 个）未读取。

## 逐项审计

| 检查 | 结果 |
|---|---|
| final-reserve group id 在本目录出现次数 | 0 |
| D0 swap pairs 中 final_reserve 行被读取进特征/评分 | 否 |
| D4-Q1 final hidden states 被读取 | 否 |
| 本轮日志/缓存中出现 final-reserve gid | 否 |
