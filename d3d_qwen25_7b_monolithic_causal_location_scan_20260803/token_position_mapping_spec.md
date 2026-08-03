# token_position_mapping_spec.md

## 三位置定义

- `R_end`: Reference Answer 正文最后一个非空白 token（offset mapping）。
- `C_end`: Candidate Answer 正文 `<answer>` 的最后一个非空白 token，不含模板附加句号。
- `D_pos`: 完整 prompt 中用于预测 continuation " A"/" B" 的最后一个 token，即 prompt_len-1。

## 约束

- 对每条输入要求 R_end < C_end < D_pos。
- 有效定位比例 >= 0.95 否则 `causal_location_execution_invalid`。
