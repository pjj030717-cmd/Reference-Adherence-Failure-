# d2_invalid_result_boundary_note.md

## D2 无效点与本轮修正的边界说明

D2（`prefix_causality_audit_invalid`）的无效点：

- D2 在完整 `Question + Reference + Candidate + Answer:` 序列上做前向，事后读取 Reference 末尾 token 的 hidden state。
- 该状态随后续 Candidate 文本的**序列总长度**产生数值变化（eager 与 sdpa 均复现；截断到前缀后与 T0 一致）。
- 因此 D2 的 R_end 不是严格"纯前缀"状态，D2 的结论被预注册协议判为无效。

D2-R1 的修正：

- 输入在 Reference Answer 最后一个 token 处**真正停止**：`prefix_input_ids = full_input_ids[:R_end+1]`。
- prefix 之后不存在 Candidate Answer、`Answer:`、generation prompt、padding 或 suffix token。
- `h_prefix[layer] = hidden_states[layer][0, prefix_len-1, :]` 只来自这个截断前向。

边界（不得越界解释）：

- 本轮的"合同"是：**相同截断输入重复运行的确定性**，以及 **T0/T1/T2 前缀 token ids 完全相同**。
- 不再要求完整长序列中早期 token 的 hidden state 与截断前向一致（这正是 D2 失效的机制）。
- D2 的正式标签 `prefix_causality_audit_invalid` 原样保留，不得覆写。
- D2 的完整序列 R_end 结果仅作"已作废的诊断"在报告中引用，不进入任何比较表、不合并指标、不作为正式证据。
