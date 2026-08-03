# hook_layer_mapping_audit.md

## 状态：未执行（因分段零扰动等价门失败）

协议 §5 要求在通过 §4 分段零扰动等价门后，审计 L18（hidden-state index 18，即 decoder block 17）forward hook 映射。§4 门失败（`segmented_execution_equivalence_invalid`），因此**未注册、未运行任何干预 hook**。

## 计划中的审计内容（协议要求，未执行）

```text
hidden_states index      = 18
decoder block index      = 17  (0-based)
hook 模块路径            = model.model.layers[17]（输出端 forward hook）
hook 输入/输出 tensor    = (1, prefix_len, 3584)，BF16
R_end 在 prefix 中的位置 = prefix_len - 1（prefix = full_ids[:R_end+1]）
```

要求（协议 §5）：

1. 阶段 P 注册 forward hook；
2. 只修改 `R_end` 一个序列位置；
3. 修改发生在 selected layer output、下一 decoder block 之前；
4. 其余 token / 层 / batch 项不变；
5. alpha=0 hook 前后 hidden states、cache、最终 logits 完全一致；
6. 输出 20 train + 20 dev 的 hook 层映射与零扰动审计。

由于等价门失败，以上步骤未执行，亦不会产生 alpha>0 的任何干预数值。

## 说明

本文件存在以完成交付清单完整性，并明确记录"未执行"状态。不伪造任何 hook 审计数据。
