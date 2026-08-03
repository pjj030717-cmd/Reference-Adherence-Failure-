# inheritance_audit.md

## 继承对账（D2-R1 Phase 0）

| 项 | 值 | 状态 |
|---|---|---|
| D0 final_label | `jar_style_sciq_data_qualification_feasible` | ✓ |
| D1 final_label | `jar_style_reference_override_behavior_feasible` | ✓ |
| D1-R final_label | `template_robust_reference_override_feasible` | ✓ |
| D2 final_label | `prefix_causality_audit_invalid`（原样保留） | ✓ |
| D0 split | train 587 / dev 195 / reserve 197（seed 20260802） | ✓ |
| Qwen revision | `a09a3545…` | ✓ |
| config/tokenizer/index 哈希 | 与 D1 一致 | ✓ |
| system prompt / user 模板 / chat template / continuations | 与 D1 一致 | ✓ |
| 基础模板 T0 | `The answer is <answer>.` | ✓ |
| dev pairs | 195 | ✓ |
| D2 dev SS 标签 | 195，与 D1 SS 行级结果逐行一致 | ✓ |
| D2 train SS 标签 | 587（D2 已审计与 D1 规格一致） | ✓ |

## 本轮模型读取范围

```text
train_model_scored = true（587 groups，仅 T0 真截断 prefix 前向）
dev_model_scored = true（195 groups）
final_reserve_model_scored = false（197 groups 禁止读取/评分/缓存/提取）
```

## 关键禁止

- 禁止加载/复用/比较 D2 的任何 hidden-state 数组（`d2_hidden_arrays_reused = false`）。
- D2 正式标签 `prefix_causality_audit_invalid` 原样保留，不得覆写。
