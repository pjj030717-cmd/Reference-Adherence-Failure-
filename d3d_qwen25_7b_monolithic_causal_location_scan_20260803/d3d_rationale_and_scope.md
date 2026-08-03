# d3d_rationale_and_scope.md — D3-D 实验定位说明

## 1. 动机

D3-M-R1 已确认两点：

1. L18 × R_end 的真实 prefix 表示可强力**预测**后续 SS 错拒（AUROC 0.922）；
2. 但在完整单体前向中，对该单点施加线性加性 patch 几乎不影响最终 A/B 判决
   （所有 alpha 配置 SS_net_gain=0）。

因此本实验不重复或扩大 `L18 × R_end` 的 alpha 搜索，而是回答唯一问题：

> reference-binding failure 的可操纵因果位置，是在读完 Reference 时（R_end）、
> 读完 Candidate 时（C_end）、还是即将输出 A/B 前（D_pos）才真正形成？

## 2. 与既有实验的关系

- **保留** D3-M-R1 结论：`L18 × R_end × 单一线性加性 patch 的选择性因果作用不足`。
- 本实验将因果位置搜索扩展到 12 个位置
  `(R_end, C_end, D_pos) × (L14, L18, L22, L26)`，每个位置在 train 内独立选择
  方向（V_mean / V_lda / V_logit）。
- 待干预 group 的选择**始终**由 D3-M-R1 冻结的真实 prefix 风险方向路由
  （`z_prefix = v_prefix · (h_true_prefix_R_end − mu_prefix_train)`），
  不读取 Candidate / SS 标签 / 完整 prompt 分数 / 干预结果。

## 3. 方法边界

- 全程完整单体前向，无 prefix KV cache / segmented execution。
- 干预形式固定：`h_patched = h + alpha × sigma_z[L,pos] × v[L,pos]`，
  alpha ∈ {-1.0, -0.5, -0.25, 0.25, 0.5, 1.0}（6 个），12 × 6 = 72 配置。
- 不训练 SFT / Reward Model；仅允许 Probe 以内的线性方向构造。
- 不改 prompt / T0 / 模型 / 精度 / batch / backend。

## 4. 禁止的宣称

本实验不主张：

- 参数知识是唯一原因；
- 已找到跨模型、跨任务的普适机制；
- 已证明 R_end 风险状态本身是唯一因果瓶颈；
- 已得到生产环境通用修复方法。
