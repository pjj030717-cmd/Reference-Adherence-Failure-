# inheritance_audit.md (D3-M)

## 六标签继承

| 实验 | 要求标签 | 实测 |
|---|---|---|
| D0 | `jar_style_sciq_data_qualification_feasible` | ✓ |
| D1 | `jar_style_reference_override_behavior_feasible` | ✓ |
| D1-R | `template_robust_reference_override_feasible` | ✓ |
| D2 | `prefix_causality_audit_invalid`（保持原样） | ✓ |
| D2-R1 | `true_prefix_reference_state_signal_localized` | ✓ |
| D3 | `segmented_execution_equivalence_invalid`（保持原样） | ✓ |

## 模型与语义核验

- revision `a09a35458c70…`；config/tokenizer/safetensors index 哈希与 D1 一致。
- T0 模板 SHA256 = `c42e1ea10a6be…`（`The answer is <answer>.`）。
- `" A"` / `" B"` continuation token ids = 362 / 425。
- teacher-forced pos = prompt_len - 1。
- synthetic 24-pair readout：order_accuracy = 24/24，ties = 0。

## 本轮执行边界

```text
D3 的 segmented_execution_equivalence_invalid 保持原样；
本轮不复用 D2 或 D2-R1 的 hidden-state 数组；
本轮不用 prefix cache、past_key_values、分段续算、截断后续算；
本轮只做完整原始输入的 monolithic forward。
```
