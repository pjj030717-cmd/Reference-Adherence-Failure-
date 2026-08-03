# swap_overlap_disclosure.md

## 数据泛化限制审计（描述性披露，不用于重抽或改变 split）

| 指标 | train | dev | train↔dev 交集/jaccard |
|---|---|---|---|
| 唯一 r_o 数 | 525 | 190 | 交 29 / jaccard 0.042 |
| 唯一 r_s 数 | 420 | 170 | 交 102 / jaccard 0.209 |
| 唯一 (r_o,r_s) 对 | 587 | 195 | — |

## swap donor 的 split 关系

| 目标 split | donor 来自 train | donor 来自 dev | donor 来自 reserve | unknown |
|---|---|---|---|---|
| train | 339 | 120 | 128 | 0 |
| dev | 105 | 45 | 45 | 0 |

## 结论
- 本披露仅用于说明 train/dev 在答案字符串与 swap donor 上的重叠程度，评估 R_end Probe 的泛化边界。
- 不据此重抽、不删除 group、不改变既有 split。
