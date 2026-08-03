# segmented_execution_spec.md

## 目的

D3 的 RBSP 干预必须在"真 prefix → cache → suffix"的分段执行中完成，而不是在完整长 prompt 中回读/修改 `R_end`。D2 已证明完整序列长度会污染早期 hidden state 的数值（见 `d2_invalid_result_boundary_note.md`）。

## 分段执行规格（D3 Phase B 审计版）

对每条 T0 四格输入：

```text
阶段 P（prefix）:
  token ids = 实际 chat-rendered full prompt 的 input_ids
  R_end = Reference Answer 正文最后一个非空白 token 的位置（offset mapping 定位）
  prefix_input_ids = full_input_ids[: R_end + 1]
  prefix_attention_mask = ones
  use_cache = true
  输出 past_key_values（DynamicCache，seq len = prefix_len）

阶段 S（suffix）:
  suffix_input_ids = full_input_ids[R_end + 1 :]
  cache_position = arange(prefix_len, prefix_len + suffix_len)
  在完整 prompt 最后位置（prompt_len - 1）读取 A/B logits
  d_raw = log p(" A" | prompt) - log p(" B" | prompt)
```

## 干预点（设计意图）

- `hidden-state index = 18`（D2-R1 选定），对应 decoder block `17` 的 output。
- 在阶段 P 对 `layer.model.layers[17]` 注册 forward hook，只修改序列位置 `R_end`（即 prefix 最后一位 `prefix_len - 1`）。
- `alpha = 0` 时 hook 恒等，前向行为不得改变。

## 零扰动等价门要求（协议 §4）

对 dev 195 × 4 = 780 个 T0 输入：

```text
predicted labels 780/780 与 D1 完全一致
l_A / l_B / d_raw 在 BF16 序列化精度下一致
OO / OS / SO / SS = 1.000 / 1.000 / 0.928 / 0.241
ACC_o = 1.000；ACC_s = 0.585；RPAG = 0.415
```

任何不满足 → `segmented_execution_equivalence_invalid`，立即停止。

## 审计结果：segmented-execution 与 monolithic 不等价

在默认后端（`_attn_implementation = sdpa`，与 D1 完全一致）下运行 dev 780 行：

| 指标 | 值 |
|---|---|
| label mismatch | **3 / 780** |
| d_raw 绝对差 mean | 0.1831 |
| d_raw 绝对差 median | 0.1250 |
| d_raw 绝对差 p95 | 0.5000 |
| d_raw 绝对差 max | 2.1250 |
| l_A/l_B 最大 BF16-ULP 偏差 | 21798.00 |
| OO label-match | 1.0000 |
| OS label-match | 1.0000 |
| SO label-match | 0.9897 |
| SS label-match | 0.9949 |

mismatch 明细：

```text
53a9a275a436 SO  D1=B  seg=A  (d1_d=-1.1875 → seg_d=+0.25)
9a48b8a8d35c SO  D1=B  seg=A  (d1_d=-0.375  → seg_d=+0.375)
bb16e19b3e8a SS  D1=A  seg=B  (d1_d=+0.5    → seg_d=-0.1875)
```

## 根因

1. **monolithic 参考可信**：3 个 mismatch 样本用完整前向重算，与 D1 CSV 逐位一致（|Δd| = 0.0），排除 D1 参考异常。
2. **分段续算 API 正确**：toy 序列（24 tokens，split=12）分段续算与完整前向 diff = 0.000000，证明 `cache_position` 续算本身数学等价；split 未对齐时出现 ~1 ULP 的 kernel 数值噪声。
3. **模型实现对序列总长度固有数值敏感**：即使 eager + 严格 causal mask，`pos0`（只 attend 自己）的 hidden state 也随序列总长度变化（layer1 起 ~0.008，随层累积至 layer28 ~1.6-23.6）；phase P 的 prefix KV 与 monolithic 中相同前缀的 KV 数值不同（key max diff = 2.0，value = 0.97）。D2-R1 `d2_invalid_result_boundary_note.md` 已记录："该状态随后续 Candidate 文本的序列总长度产生数值变化（eager 与 sdpa 均复现）"。
4. 因此：phase P 以 99-token 截断序列计算，monolithic 以 116-token 完整序列计算同一 R_end 位置时，数值必然不同，且差异在 28 层前向中放大到可翻转最终判决。

## 结论

`segmented_execution_equivalence_invalid`

按协议 §4 与 §11：不得换 backend / batch / 精度 / padding / cache 实现或提示词规避。停止一切干预实验。不进入 hook 干预、方向构造、grid 选择与 dev 因果确认。
