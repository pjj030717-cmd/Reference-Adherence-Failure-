# inheritance_audit.md — D3-M-R1 继承核验审计

## 1. 七标签核验（逐项一致）

| 实验 | 期望标签 | 实际标签 | 一致 |
|---|---|---|---|
| D0 | `jar_style_sciq_data_qualification_feasible` | 同 | ✓ |
| D1 | `jar_style_reference_override_behavior_feasible` | 同 | ✓ |
| D1R | `template_robust_reference_override_feasible` | 同 | ✓ |
| D2 | `prefix_causality_audit_invalid` | 同 | ✓ |
| D2R1 | `true_prefix_reference_state_signal_localized` | 同 | ✓ |
| D3 | `segmented_execution_equivalence_invalid` | 同 | ✓ |
| D3M | `monolithic_direction_label_capacity_insufficient` | 同 | ✓ |

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

## 3. T0 模板

- 模板字符串：`The answer is <answer>.`
- UTF-8 SHA256：`c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc`（与 D1R 一致）

## 4. D1 synthetic A/B readout

- D1 记录：24/24 正确，0 个 TIE（`synthetic_readout_audit.csv`）。

## 5. D3-M 已确认项（本轮继承，不重复推断）

- D1 的 780 个 dev T0 四格标签逐位复现（label_mismatch=0，max_bf16_ulp=0.0）。
- `hidden_states[18]` 等于 `model.model.layers[17]` 输出（L18 hook 层映射审计通过）。
- R_end 定位与 D2-R1 一致。
- 被动 hook / zero hook 780/780 等价，`Δd_raw = 0`。

## 6. D3-M 未越界确认

- D3-M 的 decision.json 确认：未拟合方向、未执行真实干预、未读取 final-reserve。

## 7. 本轮 Phase 0 回归（dev 780，单体完整前向）

- 780/780 prediction 与 D1 完全一致（label_mismatch=0）。
- max_abs_delta_d_raw = 0.0。
- hook 的 R_end state 与 `hidden_states[18][R_end]` 一致（r_end_mismatch=0）。
- 结果文件：`monolithic_hook_equivalence_audit.csv`。

**结论：继承核验全部通过，零扰动资格成立。**
