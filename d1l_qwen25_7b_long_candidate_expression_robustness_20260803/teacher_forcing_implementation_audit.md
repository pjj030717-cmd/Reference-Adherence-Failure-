# teacher_forcing_implementation_audit.md

## 正确 teacher-forced 实现（与 D1 逐字一致）

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

- 读取位置固定 `pos = prompt_len - 1`；禁止 off-by-one / 拼接 continuation 后取末位 logits。
- 无 prior 校正、无空白偏置、无温度校正、无阈值调参。
- `p_accept_raw = 1/(1+exp(-d_raw))`；`prediction = A if d_raw>0 else (B if d_raw<0 else TIE)`。
- 每模板 195 group × 4 cell = 780 次固定 A/B 判断；T0/T3/T4/T5 同一 q/r_o/r_s，仅替换 Candidate 渲染。
