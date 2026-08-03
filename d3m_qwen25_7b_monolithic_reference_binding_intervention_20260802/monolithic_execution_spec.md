# monolithic_execution_spec.md (D3-M)

## 执行方式

本轮只使用**完整原始输入的一次 monolithic forward**。每条 T0 四格输入：

```text
完整 T0 prompt（system + Question + Reference Answer + Candidate Answer + Answer:）
   -> 一次标准完整前向（model(input_ids)，batch=1，BF16，inference_mode）
   -> 在 model.model.layers[17]（hidden_states index 18）输出处
      仅修改 R_end 位置（可选干预）
   -> 同一次完整前向的后续层继续计算
   -> 在 prompt_len - 1 读取 A/B logits
   d_raw = logp(" A") - logp(" B")
```

明确不使用：`prefix cache`、`past_key_values`、`cache_position` 续算、截断后重算、分段执行。

## 关键验证结果（本轮已执行）

| 阶段 | 结果 |
|---|---|
| Phase 0B 完整前向基线 | 780/780 标签与 D1 一致；max BF16-ULP = 0.0（逐位一致） |
| Phase 0C hook 层映射 | `hidden_states[18]` = `model.model.layers[17]` 输出，R_end 处逐位一致 |
| Phase 0D 被动 hook 零干预 | 780/780 一致；max Δd_raw = 0.0；无 NaN、无 tie 增加 |

完整 monolithic 前向（含不修改输出的被动 hook）与 D1 精确等价，这是本轮与旧 D3（segmented）的本质区别。

## 干预点（设计规格）

- hidden-state index = 18（D2-R1 表示层），decoder block index = 17，模块路径 `model.model.layers[17]`。
- 干预公式（设计，本轮未实际执行干预）：

```text
h_patched = h - alpha * sigma_proj * v        # 真实方向（SS 风险降低）
h_reverse = h + alpha * sigma_proj * v        # 反方向
h_random  = h - alpha * sigma_proj * r_i      # 等范数随机方向
```

- coverage 由 D3M-fit 上冻结的风险阈值决定；选中组四格全部施加相同 patch。
- `sigma_proj = std(v·h)` 仅在 D3M-fit 上计算。

## 停止点

Phase 1 容量门失败（D3M-tune y0=39 < 40），未执行任何干预。完整 monolithic 前向与被动 hook 等价性已在本文件记录并审计通过。
