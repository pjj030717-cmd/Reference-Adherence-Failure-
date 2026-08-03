# d3m_amendment_and_rationale.md — D3-M → D3-M-R1 修正说明

## 1. D3-M 为什么停止

D3-M 在 Phase 1 构造方向时，将 587 个 train group 按协议切成
D3M-fit（70%）与 D3M-tune（30%）。tune 子集中 SS 接受（y=0）仅 39 条，
低于协议要求的 40 条阈值，触发：

```text
monolithic_direction_label_capacity_insufficient
```

D3-M 在停止前已经逐位确认了单体完整前向、L18 hook 层映射、被动 zero hook
等价性；它从未拟合方向、从未执行真实干预、从未读取 final-reserve。

## 2. 本轮（D3-M-R1）修正

协议明确禁止换 seed 规避容量问题，因此本轮**不再把 train 切成 fit/tune**，
而是直接用全部 587 个 train group 进行方向构造与 5-fold group-stratified OOF
评估。这样：

- 方向训练数据 = 全部 train（587），不存在 tune 容量不足；
- OOF 评估仍在 train 内完成，方向选择不触碰 dev / final-reserve；
- 选择完毕后用全部 587 条重拟合冻结 `v*`、`mu_train`、`sigma_z_train`。

这是 D3-M 与 D3-M-R1 的唯一实质性改动。

## 3. 固定不变的部分

- 模型 / revision / BF16 / eval / inference_mode / batch=1：不变。
- T0 模板、A/B 读出、teacher-forced 位置：不变。
- 方向特征必须是真实截断 prefix（Question+Reference，不含 Candidate/Answer:/
  generation prompt），与 D2-R1 相同：`prefix_input_ids = full_input_ids[:R_end+1]`。
- 干预点：单体完整前向中 `model.model.layers[17]` 输出、R_end 位置，
  `h_patched = h + alpha * sigma_z_train * v*`。
- 不使用 prefix cache / 分段续算 / 截断后缀重算。
- 不读取 final-reserve，直到 development 的 direction/threshold/coverage/alpha 全部冻结。

## 4. 结果

Phase 2（development 上冻结干预配置）评估了 4 个 coverage q 与 8 个 alpha 共
32 个配置，全部不满足入选条件（详见 `dev_risk_selection_and_grid.csv`），
因此：

```text
monolithic_patch_dev_selectivity_insufficient
```

本轮未产生冻结配置，未进入 Phase 3，未读取 final-reserve。
