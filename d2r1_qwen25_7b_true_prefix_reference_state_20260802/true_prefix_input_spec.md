# true_prefix_input_spec.md

## 真截断 reference-prefix 输入规格

对每个 T0 的 SS 输入：

1. 按 D1 的完整真实 prompt 与 chat template 构造 token ids：
   - system prompt（D1 固定 Judge prompt）
   - user 模板（字段顺序 Question/Reference/Candidate，逐字继承 D1）
   - `apply_chat_template(tokenize=False, add_generation_prompt=True)` → rendered prompt
   - `tok(rendered, return_offsets_mapping=True, add_special_tokens=False)` → `full_input_ids` + offsets
2. 用 offset mapping 定位 `Reference Answer` 正文最后一个非空白 token：`R_end`。
3. 构造真正的 prefix 输入：
   ```python
   prefix_input_ids = full_input_ids[: R_end + 1]
   prefix_attention_mask = ones_like(prefix_input_ids)
   ```
4. 将截断序列单独送入模型（batch size 1，`inference_mode()`，BF16）。
5. 在截断序列最后一位读取每层 hidden state：
   ```python
   h_prefix[layer] = hidden_states[layer][0, prefix_len - 1, :]
   ```

## 关键限制

- prefix 之后**不存在** Candidate Answer token、`Answer:` token、generation prompt token。
- 不得把任何 future token / padding token / suffix token 留在 attention mask 中。
- 不得用完整序列的中间位置 state 替代截断前向。
- `h_prefix` 是本轮唯一允许作为主 Probe 特征的 hidden state。

## 与 D2 的差异（唯一修改）

- D2：完整序列前向，事后读取 Reference 末尾 state（随序列总长度数值漂移）。
- D2-R1：输入在 Reference 末尾**真正停止**，截断前向的最后一位 state。
