# D3-M-R1 final_report — 单体前向下的 Reference-Binding 方向构造与选择性因果干预

## 一、结果摘要

| 问题 | 结果 |
|---|---|
| D3-M 为什么停止？ | train 切成 D3M-fit/D3M-tune 后，tune 中 SS 接受（y=0）仅 39 < 40，触发 `monolithic_direction_label_capacity_insufficient` |
| 本轮是否改变 D3-M 的 split seed？ | 否（本轮不再把 train 切成 fit/tune，直接用全部 587 组构造方向；split seed 从未被改变） |
| 方向是否只用 train 的真实 prefix 状态构造？ | 是（Question+Reference 真截断 prefix，L18/R_end；与 D2-R1 存储逐位一致，max_abs_diff=0.0；标签来自完整 T0-SS 单体前向，与 D3-M/D2-R1 各 587/587 一致） |
| dev 是否在读取 final 前冻结全部配置？ | 无配置可冻结——32 个 (q, alpha) 配置全部不合格，故从未进入"冻结→读取 final"流程 |
| final-reserve 是否只读取一次？ | 未读取（Phase 2 无合格配置即停止，final-reserve 全程零接触） |
| 单体 zero-equivalence 是否通过？ | 是（dev 780/780 与 D1 逐位一致，max_abs_delta_d_raw=0.0，R_end 定位与 D2-R1 一致） |
| real patch 的 SS 改善 | 0（SS_net_gain=0.0，干预未翻转任何预测） |
| real patch 的非 SS 伤害 | 0（nonSS_added_harm=0.0） |
| real patch 是否超过 reverse / random controls？ | 未评估（无合格配置，未进入 Phase 3 controls） |
| 最终标签 | `monolithic_patch_dev_selectivity_insufficient` |

## 二、实验结论（允许的表述）

> 在本冻结的 Qwen2.5-7B、SciQ JAR-style reference-binding setting 中，一个由
> 真实 reference-prefix 状态学习、并在单体前向 R_end 处实施的线性方向干预，
> 在本设定的 alpha 范围内无法产生足以翻转任何 Judge 预测的因果作用，
> 因此未能达到选择性减少 SS reference-binding failure 的开发门槛；
> dev 阶段即判为选择性不足，未进入 final-reserve 一次性确认。

不得主张（本轮明确不成立）：

- 已证明参数知识是唯一原因；
- 已发现普适的全局机制方向；
- 已在多模型/多领域中证明有效；
- 已获得生产环境可部署修复；
- D2 的完整序列 R_end 状态是严格因果前缀状态。

## 三、执行链路

### Phase 0：继承与零扰动资格（通过）

- 七标签核验（D0/D1/D1R/D2/D2R1/D3/D3M）全部一致。
- 模型哈希、T0 模板、synthetic 24/24、D3-M 结论继承均核验通过
  （详见 `inheritance_audit.md`）。
- dev 780 单体完整前向 zero-equivalence 回归：
  780/780 与 D1 一致，max_abs_delta_d_raw=0.0，hook R_end 与 hidden_states[18]
  一致（`monolithic_hook_equivalence_audit.csv`）。

### Phase 1：用全部 train 构造方向（完成并冻结）

- 特征：train 587 组真截断 prefix 的 L18/R_end hidden state（重提取，
  与 D2-R1 存储逐位一致）；标签 y（SS 错拒/接受）来自完整 T0-SS 单体前向。
- 5-fold group-stratified OOF 候选：V_mean / V_lda / V_logit(C∈{0.001,0.01,0.1})。

| method | AUROC | AUPRC | balacc |
|---|---|---|---|
| V_mean | 0.8906 | 0.9647 | 0.8038 |
| V_lda | 0.8998 | 0.9692 | 0.8297 |
| V_logit@0.001 | 0.9217 | 0.9768 | 0.8346 |
| **V_logit@0.01** | **0.9223** | **0.9777** | **0.8537** |
| V_logit@0.1 | 0.9207 | 0.9772 | 0.8537 |

- 选择：V_logit@C=0.01（最大 OOF AUPRC 0.9777，且与 D2-R1 选定方向
  L18/C=0.01 一致，独立交叉验证）。
- 冻结：v*、mu_train、sigma_z_train=2.8035（train 上 z 的类间效应 2.19σ）。
- 产物：`direction_candidate_oof_metrics.csv`、`direction_selection_audit.md`、
  `frozen_direction_artifact.npz`、`frozen_direction_metadata.json`、
  `train_label_capacity_audit.csv`、`true_prefix_hidden_manifest.json`。

### Phase 2：dev 冻结干预配置（无合格配置，停止）

- 风险选择：dev 195 组真截断 prefix → z_dev = v*·(h_prefix - mu_train)；
  q ∈ {1.00, 0.75, 0.50, 0.25} 取最高 z 的组，选中组的四格全部干预。
- 干预：单体完整前向，layers[17] 输出 R_end，h_patched = h + alpha·sigma_z·v*，
  alpha ∈ {-2,-1,-0.5,-0.25,0.25,0.5,1,2}。
- 32 个配置全部不合格：

| q | 选中组数 | selected_SS_base_error | SS_net_gain | nonSS_added_harm | CSI |
|---|---|---|---|---|---|
| 1.00 | 195 | 148 | 0.0 | 0.0 | 0.0 |
| 0.75 | 146 | 136 | 0.0 | 0.0 | 0.0 |
| 0.50 | 98 | 93 | 0.0 | 0.0 | 0.0 |
| 0.25 | 49 | 48 | 0.0 | 0.0 | 0.0 |

- 无任何配置满足 `selected_SS_base_error_count>=20 && SS_net_gain>=0.10 &&
  nonSS_added_harm<=0.02 && CSI>=0.08` → 立即停止，输出
  `monolithic_patch_dev_selectivity_insufficient`。
- 产物：`dev_risk_selection_and_grid.csv`、`dev_configuration_freeze.json`。

### Phase 3：final-reserve（未执行）

未读取、未评分、未提取 final-reserve。相关文件为 NOT RUN 占位。

## 四、因果效应弱的机制证据

1. hook 写入验证：注入 delta 后 L18/R_end 差分范数等于 |delta|（280.34），
   写入成功。
2. 传播验证：L18/R_end 置零后 hidden_states[19][R_end]=102.8，后续层改变。
3. 阳性对照：同 hook 机制用于最后一层最后位置 +10，
   d_raw 从 -3.0 翻转到 +0.156，证明 hook 返回值被模型采用。
4. alpha=±2（|delta|=5.61）仅使 d_raw 移动 ~0.1；最接近边界（d_raw=-0.75）
   的 SS 错拒组在 alpha=-2 时仍为 -0.6875。
5. 极端干预（R_end := 0 或 :=100）也只移动 d_raw ~0.9，
   而 SS 错拒组 d_raw 分布为 -0.75 ~ -19.78（中位 ≈ -15）。

方向在状态空间可预测（AUROC 0.92）但加性干预因果传导被后续层大幅稀释。
结论限定为：该具体方法（完整前向、L18×R_end、线性单方向、加性 patch）
不具备选择性因果作用；这是 D2-R1"指示性信号"与强因果杠杆之间的真实落差。

## 五、交付文件清单

全部产物位于
`d3mr1_qwen25_7b_monolithic_prefix_direction_intervention_20260802/`（本报告所在目录，仓库内）。

已生成：final_report.md / inheritance_audit.md / d3m_amendment_and_rationale.md /
model_access_audit.md / true_prefix_direction_input_spec.md /
true_prefix_hidden_manifest.json / train_label_capacity_audit.csv /
direction_candidate_oof_metrics.csv / direction_selection_audit.md /
frozen_direction_artifact.npz / frozen_direction_metadata.json /
dev_risk_selection_and_grid.csv / dev_configuration_freeze.json /
monolithic_hook_equivalence_audit.csv / failure_examples.md /
artifacts/decision.json；Phase 3 四文件为 NOT RUN 占位。
