# tokenization_and_teacher_forcing_audit.md

## A/B continuation（Qwen tokenizer）

| 项 | " A" | " B" |
|---|---|---|
| encode() | [362] | [425] |
| 单 token | 是 | 是 |
| 等长 | 1 | 1 |
| UNK | 否 | 否 |

## 无 Reference 二选一 prompt（固定）

- system：`Choose the option that is factually correct for the question.\nReply with only A or B.`
- user 模板：`Question:\n{q}\n\nOption A:\n{option_a}\n\nOption B:\n{option_b}\n\nAnswer:`
- 无 Reference / Candidate / Judge / Accept / Reject 字样。

## teacher-forced 位置

```python
pos = prompt_len - 1
d_raw = l_A - l_B   # l_A = log p(" A"), l_B = log p(" B")
```

- Order1: A=r_o, B=r_s -> d_1 = l_A - l_B
- Order2: A=r_s, B=r_o -> d_2 = l_B - l_A（以 r_o 为偏好方向）
- k = (d_1 + d_2) / 2
- 无空白先验校正 / 阈值调参 / prompt 搜索。
