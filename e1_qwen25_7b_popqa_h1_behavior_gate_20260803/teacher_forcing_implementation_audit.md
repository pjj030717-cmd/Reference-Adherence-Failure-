# teacher_forcing_implementation_audit.md

## 正确 teacher-forced 实现（与 D1 一致）

```python
prompt_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
prompt_len = prompt_ids.shape[1]
logits = model(prompt_ids).logits
pos = prompt_len - 1
logits_last = logits[0, pos, :]
l_A = logits_last[362]   # " A"
l_B = logits_last[425]   # " B"
d_raw = l_A - l_B
```

- 读取位置固定 `pos = prompt_len - 1`；禁止 off-by-one / 拼接 continuation 后取末位 logits。
- 无 prior 校正、无空白偏置、无温度校正、无阈值调参、无自由生成。
- 对每个 dev group：OO/OS/SO/SS 四格固定 A/B 判断；`prediction = A if d_raw>0 else (B if d_raw<0 else TIE)`。
- 仅最终层 logits；不提取 hidden state。
