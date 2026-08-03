# failure_examples.md

## segmented-zero 等价门失败的典型样本

审计方法：对 dev 195 × 4 个 T0 输入，`segmented-zero`（phase P 99-token 真截断 prefix + cache 续算）与 D1 monolithic 逐行比较。

| source_group_id | cell | D1 pred | seg pred | D1 d_raw | seg d_raw | Δd_raw |
|---|---|---|---|---|---|---|
| 53a9a275a436… | SO | B | A | -1.1875 | +0.2500 | +1.4375 |
| 9a48b8a8d35c… | SO | B | A | -0.3750 | +0.3750 | +0.7500 |
| bb16e19b3e8a… | SS | A | B | +0.5000 | -0.1875 | -0.6875 |

这三条是全部 780 行中仅有的标签翻转样本。它们恰好落在 d_raw ≈ 0 的判决边界附近，因此 kernel 数值噪声（见下）足以翻转预测。

## 根因示例：pos0 的 hidden state 随序列总长度变化

即使 eager 注意力 + 严格 causal mask（pos0 只 attend 自己），在 99-token 截断序列与 116-token 完整序列中：

```text
pos0  layer1  max diff = 0.0078
pos5  layer1  max diff = 0.1563
pos98 layer28 max diff = 23.5625
```

phase P 前缀 KV 与 monolithic 前缀 KV 的数值差异：

```text
key max diff   = 2.0000
value max diff = 0.9688
```

两次独立 phase P 前向完全一致（diff = 0.0），即差异是确定性的、可复现的，不是随机噪声。

## 结论

这不是分段续算 API 用法错误（toy 24-token split=12 精确等价），而是模型实现对"序列总长度"的固有数值敏感，导致真截断 prefix 前向与完整序列前向在同一 R_end 位置的 hidden state 数值不同。该现象与 D2 记录一致，构成 `segmented_execution_equivalence_invalid`。
