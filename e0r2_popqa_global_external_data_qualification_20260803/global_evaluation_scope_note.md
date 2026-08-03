# global_evaluation_scope_note.md

## 总体外部复现集的研究定位

- 本研究单位：source group（共 14,077）。
- 主指标：全部 group 的 micro-average（后续 H1）。
- relation/property 的作用：仅作为 **donor matching constraint**，确保 r_o 与 r_s 的答案交换不跨越粗粒度关系类型；
  16 类 relation 不被当作 16 个独立实验任务。
- relation 分层只允许描述性诊断：
  - `n >= 30` 的 relation 可报告带 95% CI 的描述性数值；
  - `n < 30` 的 relation 只报告样本数，不下 relation-specific 结论。
- 稀有 relation（如 color：dev 仅 4 组）不删除、不重采样、不单独挑选、不重切分。

## 每 split relation 统计（描述性）

| split | relation 数 | 最小类样本数 | 最小类 | 最大类占比 | 最大类 |
|---|---|---|---|---|---|
| train | 16 | 17 | color | 0.142908 | screenwriter |
| dev | 16 | 4 | color | 0.148845 | director |
| final_reserve | 16 | 13 | color | 0.142401 | screenwriter |

## 达标 relation（n >= 30，未来可报告带 CI 的分层数值）

基于每 split 计数：除 dev 的 `color`（4 组）与 final-reserve 的 `color`（13 组）外，其余 relation 在各 split 均 >= 17。
train split 全部 16 类 >= 17。
