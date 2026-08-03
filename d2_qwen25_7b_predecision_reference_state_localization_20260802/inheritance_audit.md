# inheritance_audit.md

## 继承对账（Phase 0）

| 项 | 值 | 状态 |
|---|---|---|
| D0 final_label | `jar_style_sciq_data_qualification_feasible` | ✓ |
| D1 final_label | `jar_style_reference_override_behavior_feasible` | ✓ |
| D1-R final_label | `template_robust_reference_override_feasible` | ✓ |
| D0 split | train 587 / dev 195 / reserve 197（seed 20260802） | ✓ |
| Qwen revision | `a09a3545…` | ✓ |
| config/tokenizer/index 哈希 | 与 D1 model_access_audit.md 一致 | ✓ |
| system prompt / user 模板 / chat template / continuations `" A"`/`" B"` | 与 D1 一致 | ✓ |
| 基础模板 T0 | `The answer is <answer>.`（SHA256 `c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc`） | ✓ |
| dev pairs | 195（与 D0 dev ids 一致） | ✓ |
| D1 四格行 | 780（four_cell_scores_dev.csv） | ✓ |

## 本轮模型读取范围

```text
train_model_scored = true（587 groups）
dev_model_scored = true（195 groups）
final_reserve_model_scored = false（197 groups 禁止读取/评分/缓存/提取）
```

## 备注

- D0 中间态 hash 缺陷按 D1-R `provenance_amendment.md` 处理，不修改 D0。
