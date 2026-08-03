# inheritance_audit.md

## 继承对账（D3 Phase 0）

| 项 | 值 | 状态 |
|---|---|---|
| D0 final_label | `jar_style_sciq_data_qualification_feasible` | ✓ |
| D1 final_label | `jar_style_reference_override_behavior_feasible` | ✓ |
| D1-R final_label | `template_robust_reference_override_feasible` | ✓ |
| D2 final_label | `prefix_causality_audit_invalid` | ✓ |
| D2-R1 final_label | `true_prefix_reference_state_signal_localized` | ✓ |
| D0 split | train 587 / dev 195 / reserve 197 | ✓ |
| Qwen revision | `a09a3545…` | ✓ |
| config/tokenizer/index 哈希 | 与 D1 一致 | ✓ |
| system prompt / user 模板 / chat template / continuations | 与 D1 一致 | ✓ |
| 基础模板 T0 | `The answer is <answer>.` | ✓ |
| D2-R1 selected layer/C | L18 / C=0.01（RBSP 继承） | ✓ |
| D2-R1 SS 评分表 | dev 195 / train 587 | ✓ |
| D2-R1 h_prefix | (28, 3584)，允许继承用于方向构造 | ✓ |
| D1 four-cell dev | 780 行 | ✓ |

## 本轮模型读取范围

```text
train_model_scored = true（587 groups，分段执行）
dev_model_scored = true（195 groups）
final_reserve_model_scored = false（197 groups 禁止）
```

## 关键继承约束

- RBSP 方向构造只允许使用 D2-R1 的 true-prefix `L18/R_end` states（train 内 D3-fit 子集）。
- D3-fit / D3-tune 只在 train 587 内以 seed 20260802 分层切分（70/30），group 四格同子集。
- dev 不得用于拟合方向、选择 alpha、选择触发阈值。
