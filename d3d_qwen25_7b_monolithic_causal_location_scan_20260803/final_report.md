# D3-D final_report — Reference-Binding Failure 的单体前向因果位置定位与选择性干预

## 一、结果摘要

| 问题 | 结果 |
|---|---|
| D3-M-R1 的单点 R_end patch 结论是否保留？ | 是（`L18 × R_end × 单一线性加性 patch 的选择性因果作用不足` 原样保留） |
| 本轮是否使用 segmented execution？ | 否（全程完整单体前向，无 prefix KV cache） |
| 三个 token 位置是否均可机械定位？ | 是（dev 780 全部 R_end < C_end < D_pos，offset 唯一，有效定位 100%） |
| 12 个位置是否均通过 zero-equivalence？ | 是（被动 hook 780/780 与 D1 逐位一致，max_abs_delta_d_raw=0.0） |
| 待干预 group 是否只由真实 prefix 风险选择？ | 是（仅用 D3-M-R1 冻结的 V_logit@C=0.01 prefix 方向 + train z 中位数阈值 0.6158；dev 选中 97 组，其中 92 组 SS 基线错误） |
| dev 是否在读取 final 前冻结唯一配置？ | 无配置可冻结——72 个 (位置, alpha) 配置全部不合格，未进入"冻结→读取 final"流程 |
| 最佳因果位置 | 无（72 配置中最大 CSI=0.0103（L18/C_end, alpha=0.25），门槛 0.13） |
| real 是否优于 reverse / random controls？ | 未评估（无合格配置，未进入特异性控制） |
| final-reserve 是否只读取一次？ | 未读取（dev 门失败即停止，final-reserve 全程零接触） |
| 最终标签 | `causal_location_dev_not_found` |

## 二、实验结论（允许的表述）

> 在本冻结的 Qwen2.5-7B、SciQ JAR-style reference-binding setting 中，对
> R_end / C_end / D_pos 三个时间位置（各 × L14/L18/L22/L26）进行单点、线性、
> 加性 patch 干预，在固定 alpha 范围内均无法翻转任何 Judge 预测，未发现可操纵
> 的选择性因果位置；dev 阶段即判定因果位置未找到，未进入 final-reserve 一次性
> 确认。

不得主张（本轮明确不成立）：

- 已证明参数知识是唯一原因；
- 已找到跨模型、跨任务的普适机制；
- 已证明 R_end 风险状态本身是唯一因果瓶颈；
- 已得到生产环境通用修复方法。

## 三、执行链路

### Phase 0：继承、位置定位与零扰动核验（全部通过）

- 七标签核验一致；模型哈希一致；D3-M-R1 未读 final-reserve 且 prefix 方向可读。
- dev 780 三位置机械定位：全部 R_end < C_end < D_pos，offset 唯一（100% 有效）。
- 被动 hook 零扰动：dev 780 全部 12 位置，prediction 与 D1 一致（780/780），
  max_abs_delta_d_raw=0.0。
- `hidden_states[L]` 与 `layers[L-1]` 输出映射逐位验证一致。
- 产物：`token_position_mapping_spec.md`、`token_position_mapping_audit.csv`、
  `passive_hook_zero_equivalence_audit.csv`、`inheritance_audit.md`、
  `model_access_audit.md`。

### Phase 1：train 内构造 12 位置方向（完成并冻结）

- train 587 组 SS 完整输入，一次单体前向提取
  `(R_end, C_end, D_pos) × (L14, L18, L22, L26)` 12 位置 hidden state。
- 每位置独立 5-fold OOF 选择方向（固定规则），冻结 `v[L,pos]`, `mu`, `sigma_z`。
- OOF 预测梯度：R_end 0.85–0.93 < C_end 0.89–0.97 < D_pos 0.90–0.99996。
- 产物：`train_location_direction_oof_metrics.csv`、
  `train_location_direction_selection.json`、`train_location_direction_artifact.npz`。

### Phase 2：固定风险路由（通过）

- `v_prefix` = D3-M-R1 V_logit@C=0.01；`z_prefix = v_prefix·(h_true_prefix_R_end − mu_prefix)`。
- t_prefix = train z 中位数 = 0.6158。
- dev 选中 97/195 组（约 50% 覆盖率），其中 92 组为 SS 基线错误。
- 选中组全部四格施加同一 patch，未选中组仅用于覆盖率报告。
- 产物：`frozen_prefix_risk_selection_spec.json`、`dev_selected_group_manifest.csv`。

### Phase 3：dev 因果位置扫描（无合格配置，停止）

- 72 配置（12 位置 × 6 alpha）全部在 selected group 上评估。
- 入选门槛：selected_SS_base_error_count>=30（满足，92）、SS_net_gain>=0.15
  （未满足，全部 ≤0.0103）、nonSS_added_harm<=0.02（满足）、CSI>=0.13（未满足）。
- 无配置合格 → 立即输出 `causal_location_dev_not_found`，未进入特异性控制，
  未读取 final-reserve。
- 产物：`dev_location_alpha_scan.csv`、`dev_configuration_freeze.json`、
  `failure_examples.md`。

### Phase 4：final-reserve（未执行）

未读取、未评分、未提取。相关文件为 NOT RUN 占位。

## 四、机制诊断摘要

1. 干预机制已逐项验证：hook 写入精确（delta 范数匹配）、传播到后续层、
   方向符号正确（负 alpha 推向接受）。
2. selected SS 错拒组的 d_raw 分布 -19.78 ~ -7.69，远离决策边界；
   单点线性加性 patch 最大移动约 +2.9（L26/D_pos），无法跨过 d_raw=0。
3. D_pos 状态几乎完全决定判决（预测 AUROC≈1.0），但加性移动它不足以扭转
   判决——最终 A/B 由深层多步计算固化。
4. 因此排除"只是位置没找对"的替代解释；R_end / C_end / D_pos 三位置在该
   干预框架下均无可操纵的选择性因果作用。

## 五、交付文件清单

全部产物位于
`d3d_qwen25_7b_monolithic_causal_location_scan_20260803/`（本报告所在目录，仓库内）。

已生成：final_report.md / inheritance_audit.md / d3d_rationale_and_scope.md /
model_access_audit.md / token_position_mapping_spec.md /
token_position_mapping_audit.csv / passive_hook_zero_equivalence_audit.csv /
train_location_direction_oof_metrics.csv / train_location_direction_selection.json /
frozen_prefix_risk_selection_spec.json / dev_location_alpha_scan.csv /
dev_selected_group_manifest.csv / dev_configuration_freeze.json / failure_examples.md /
artifacts/decision.json；dev_specificity_controls.csv 及 Phase 4 四文件为 NOT RUN 占位。
