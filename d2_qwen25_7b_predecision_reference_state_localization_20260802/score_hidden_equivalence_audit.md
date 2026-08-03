# score_hidden_equivalence_audit.md

以 `output_hidden_states=True` 的完整前向重算 dev 780 行评分，与 D1 `four_cell_scores_dev.csv` 逐行对齐：

| 核对项 | 结果 |
|---|---|
| predicted_label | 780/780 一致 |
| l_A/l_B/d_raw | BF16 序列化精度一致（max abs diff < 1e-3） |
| OO accuracy | 1.000 |
| OS accuracy | 1.000 |
| SO accuracy | 0.928 |
| SS accuracy | 0.241 |
| 四格 aggregate | 与 D1 完全一致 |
| 结论 | 含 hidden states 的前向未改变 Judge 行为 |

复现 detail 见 collection 日志（dev reproduction audit PASSED）。
