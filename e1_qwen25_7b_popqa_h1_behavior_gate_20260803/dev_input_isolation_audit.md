# dev_input_isolation_audit.md

## 静默 split filter 审计

- 来源：`E0-R1 external_swap_pairs.jsonl`（全 split 14077 行）。
- 过滤方式：仅依据每行 `split` 字段机械判断；非 dev 行立即丢弃。
- 保留：dev 2,815 group 的文本输入，写入 `scripts/_dev_input.jsonl`。

## 隔离标志

```text
source_stream_scanned_for_split_filter = true
final_reserve_text_exposed_to_model = false
final_reserve_model_scored = false
final_reserve_hidden_state_read = false
train_text_exposed_to_model = false
```

## 遵守约束

1. 只依据已有 `split` 字段 / 冻结 group-id manifest 过滤：✓
2. 不打印、保存或抽样非 dev 文本：✓（丢弃行仅计数 {'train': 8446, 'final_reserve': 2816}）
3. 不对 non-dev 文本做 tokenization / 前向 / 评分 / 任何统计：✓
4. 过滤完成后仅保留 dev 2,815 group 文本输入：✓
5. dev manifest（group_count / sorted_group_id_sha256 / relation_distribution_sha256）与
   E0-R2 approved manifest 完全一致：✓
