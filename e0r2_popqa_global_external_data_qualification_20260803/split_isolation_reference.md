# split_isolation_reference.md

## split 隔离（E0/E0-R1 已核验，E0-R2 引用）

- split seed：20260816；先 dict 排序 source_group_id，再 `random.Random(20260816).shuffle`；60/20/20。
- train / dev / final-reserve = 8,446 / 2,815 / 2,816。
- 每个 source_group_id 仅属于一个 split；跨 split 零重叠（14,077 个唯一 group）。
- donor 选择仅在 split 内部进行：同 split、同 relation、不同 source_group_id。
- fixed split 索引见 E0 `fixed_split_indices.json`。
