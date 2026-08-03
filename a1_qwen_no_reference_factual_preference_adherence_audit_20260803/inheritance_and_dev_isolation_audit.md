# inheritance_and_dev_isolation_audit.md

## 既有结论（只读核验）

| 实验 | 标签 | 状态 |
|---|---|---|
| SciQ groups == 195 | 195 | ✓ |
| SciQ 4 cells each |  | ✓ |
| SciQ r_o != r_s | 0 violations | ✓ |
| SciQ y_SS=1 count == 148 | 148 | ✓ |
| PopQA groups == 2815 | 2815 | ✓ |
| PopQA dev input groups match fourcell groups |  | ✓ |
| PopQA r_o != r_s | 0 violations | ✓ |
| PopQA y_SS=1 count == 144 | 144 | ✓ |
| PopQA 16 relations | 16 | ✓ |
| PopQA dev manifest sha == E0-R2 approved | 14aa6be52f0698aa… vs 14aa6be52f0698aa… | ✓ |
| synthetic manifest 24 pairs (12 A / 12 B) |  | ✓ |
| prompt spec written + sha | dc55c01dcf9c5d7e… | ✓ |

## dev 隔离

```text
source_stream_scanned_for_split_filter = false (PopQA dev 直接取自 E1 已隔离的 _dev_input.jsonl)
final_reserve_text_exposed_to_model = false
final_reserve_model_scored = false
train_text_exposed_to_model = false
hidden_states_read = false
probe_trained = false
activation_intervention_run = false
```

- SciQ dev：D1 `four_cell_scores_dev.csv`（195 group，SS 标签逐字继承）。
- PopQA dev：E1 `scripts/_dev_input.jsonl` + `scripts/_dev_fourcell_rows.csv`（2,815 group，SS 标签逐字继承）。
- 未读取任何 train / final-reserve 文本；未读取既有 final-reserve score/prediction 文件。

## 合成 manifest

- 24 条基础事实二选一（12 条正确在 A，12 条正确在 B），在真实 dev 评分前冻结写入
  `synthetic_factual_choice_manifest.json`；不得按模型输出修改。
