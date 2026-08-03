# teacher_forcing_implementation_audit.md

## 正确 teacher-forced 实现

对每条 prompt：

```python
prompt_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
prompt_len = prompt_ids.shape[1]
logits = model(prompt_ids).logits
pos = prompt_len - 1
logits_last = logits[0, pos, :]
l_A = logits_last[ACCEPT_ID]   # id 362 = " A"
l_B = logits_last[REJECT_ID]   # id 425 = " B"
d_raw = l_A - l_B
```

- 读取位置固定为 `pos = prompt_len - 1`（add_generation_prompt 后 prompt 末尾为 assistant 起始）。
- 禁止 `pos = len(prompt) + len(continuation) - 1`；禁止拼接 continuation 后在末尾取 logits；无 off-by-one。
- 无 prior 校正、无空白偏置、无温度校正、无阈值调参。
- `p_accept_raw = sigmoid(d_raw) = 1/(1+exp(-d_raw))`。

## greedy 诊断

- 使用同位置 logits 的 argmax token 作为 greedy 首 token；解码去空白后判定方向 A/B。
- 要求每条样本 greedy 方向与 d_raw 类别预测一致（已通过 24/24）。

## 空白 prompt 诊断

- 空白 prompt（question/reference/candidate 全空）下的 `l_A - l_B`：
  `d_raw = -9.34375000`（仅诊断记录，不用于校正，不是失败条件）。
