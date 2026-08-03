# hook_layer_mapping_audit.md (D3-M)

## 层映射

- D2-R1 表示层：`hidden-state index = 18`。
- 本模型映射：`hidden_states[18]` = `model.model.layers[17]`（decoder block 17）的 forward 输出。
- decoder layer 输出为纯 tensor（非 tuple）。
- R_end 位置由 offset mapping 定位（prefix 构造与 D2-R1 合同一致，但本轮只读完整前向）。

## 唯一性验证（output_hidden_states vs 被动 hook 捕获）

| split | group | r_end/seq | max_abs_diff | bit_identical |
|---|---|---|---|---|
| dev | 05677026bd7a | 98/116 | 0.000e+00 | True |
| dev | 06efe48e5f13 | 102/120 | 0.000e+00 | True |
| dev | 0a10ac3133a9 | 106/123 | 0.000e+00 | True |
| dev | 0e5e53a12a35 | 113/132 | 0.000e+00 | True |
| dev | 0e6720dd58fc | 100/119 | 0.000e+00 | True |
| train | 004c1d1f6c7e | 101/119 | 0.000e+00 | True |
| train | 015c326e1c5b | 116/134 | 0.000e+00 | True |
| train | 01992b99d529 | 104/122 | 0.000e+00 | True |
| train | 01c2ae2d0962 | 112/130 | 0.000e+00 | True |
| train | 0240f2ad868b | 103/125 | 0.000e+00 | True |

## 结论

hidden_states[18] 与 layers[17] output 在 R_end 处逐位一致 = True。
hook 干预点 = model.model.layers[17] output（hidden_states index 18），唯一映射有效。
