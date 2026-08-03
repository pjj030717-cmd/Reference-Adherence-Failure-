# inheritance_audit.md — D3-D 继承核验审计

## 1. 七标签核验（逐项一致）

| 实验 | 期望标签 | 实际标签 | 一致 |
|---|---|---|---|
| D0 | `jar_style_sciq_data_qualification_feasible` | 同 | ✓ |
| D1 | `jar_style_reference_override_behavior_feasible` | 同 | ✓ |
| D1R | `template_robust_reference_override_feasible` | 同 | ✓ |
| D2R1 | `true_prefix_reference_state_signal_localized` | 同 | ✓ |
| D3 | `segmented_execution_equivalence_invalid` | 同 | ✓ |
| D3M | `monolithic_direction_label_capacity_insufficient` | 同 | ✓ |
| D3MR1 | `monolithic_patch_dev_selectivity_insufficient` | 同 | ✓ |

## 2. 模型访问一致性（与 D1 逐位一致）

revision = `a09a35458c702b33eeacc393d103063234e8bc28`

| 文件 | SHA256 |
|---|---|
| config.json | 7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c |
| tokenizer.json | c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 |
| tokenizer_config.json | 5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583 |
| vocab.json | ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 |
| merges.txt | 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 |
| model.safetensors.index.json | 624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028 |

## 3. D3-M-R1 继承

- 标签 `monolithic_patch_dev_selectivity_insufficient`（无合格 dev 配置）。
- **final-reserve 未读取**：`final_reserve_model_scored=False`，
  `final_reserve_hidden_states_read=False`。
- 冻结 prefix 风险方向可读取：`V_logit@C=0.01`，`frozen_direction_artifact.npz`
  中 `v_raw`(3584,)、`mu_train`(3584,)、`sigma_z_train`。本轮只用作固定风险路由，
  不再重新选择方向。

## 4. D1 复现

- D1 synthetic readout：24/24（D1 记录）。
- dev 780 四格标签可由完整单体前向复现（见 Phase 0，prediction_mismatch=0）。

## 5. hidden_states[L] ~ layers[L-1] 映射

- 对候选层 L14/L18/L22/L26，在 dev 首个 SS 输入上逐位对比
  `hidden_states[L][D_pos]` 与 `layers[L-1]` 输出捕获值：
  max abs diff = 0.0（全部一致）。

## 6. 结论

继承核验全部通过，允许进入后续阶段。
