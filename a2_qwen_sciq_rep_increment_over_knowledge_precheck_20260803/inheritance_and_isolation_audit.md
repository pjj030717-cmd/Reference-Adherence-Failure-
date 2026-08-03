# inheritance_and_isolation_audit.md

## 既有结论（只读核验）

| 实验 | 标签 | 状态 |
|---|---|---|
| D0 | jar_style_sciq_data_qualification_feasible | ✓ |
| D1 | jar_style_reference_override_behavior_feasible | ✓ |
| D1-R | template_robust_reference_override_feasible | ✓ |
| D1-L | long_candidate_expression_robust | ✓ |
| D2-R1 | true_prefix_reference_state_signal_localized | ✓ |
| D4-Q1 | qwen_true_prefix_monitor_final_confirmed | ✓ |
| A1 | no_reference_factual_preference_association_supported | ✓ |

## split / 隔离核验

| 检查 | 结果 |
|---|---|
| D2-R1 train SS rows == 587 | 587 |
| D2-R1 dev SS rows == 195 | 195 |
| D0 train split sha matches D2-R1 train gids | 167d547f08659a57… |
| D0 dev split sha matches D2-R1 dev gids | 8be6f6f3450376cb… |
| train/dev zero overlap | 0 overlapping gids |
| train_union_dev == 782 | 782 |
| zero final-reserve gid in A2 dir (all 64-hex ids in train_union_dev) | 0 foreign ids |
| M_rep dev AUROC reproduces 0.9139 | 0.913887 (recorded 0.9138872915468661) |
| M_rep dev AUPRC reproduces 0.9632 | 0.963219 (recorded 0.9632189056523878) |
| B_surface dev AUROC reproduces 0.6208 | 0.620759 (recorded 0.6207590569292697) |
| B_surface dev AUPRC reproduces 0.8182 | 0.818192 (recorded 0.8181915706506249) |
| train metadata unique | 587 |
| dev metadata unique | 195 |
| train r_o != r_s | 0 violations |
| dev r_o != r_s | 0 violations |
| no NaN/inf in rho | 0 bad |

## 隔离约束

- train=587 / dev=195 为本轮唯一合法数据；final_reserve=197 绝不读取、不评分、不缓存。
- 本目录所有 64-hex group id 均在 train_union_dev (782) 中；final-reserve gid 出现 0 次。
- 未新提取 hidden states（复用 D2-R1 已存 `prefix_hidden_states/train_*.npz` / `dev_*.npz`）。
- 未读取 D4-Q1 `prefix_hidden_states/*`（final-reserve 196 个 npz）与任何 `_final_*` 工件。
- 未读取任何 PopQA 文本或分数。
- 未运行 Judge 四格、prompt 搜索、intervention/hook。
