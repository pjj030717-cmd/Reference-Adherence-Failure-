# D2：Qwen2.5-7B 的 pre-decision reference-state 表示定位

| 问题 | 结果 |
|---|---|
| D0/D1/D1-R 是否被唯一继承？ | 是 |
| 含 hidden states 的前向是否逐行复现 D1？ | 是（dev 780/780，aggregate 1.000/1.000/0.928/0.241） |
| R_end/C_end/D_pos 是否定位有效？ | 是（offset-mapping，60 组抽查 ids 逐输入一致） |
| SS error 的 train/dev 容量是否足够？ | 是（train 468/119；dev 148/47） |
| pre-decision R_end Probe 是否优于 B_surface？ | 是（ΔAUPRC CI lower>0） |
| 是否迁移到 T1/T2？ | 行为层可复现 D1-R（T1 0.169 / T2 0.113），但 **R_end 数值一致性审计失败**（T1 R_end ≠ T0） |
| 是否读取/评分 final-reserve？ | 否 |
| 是否运行 activation intervention？ | 否 |
| 是否允许进入 D3？ | 否 |
| 最终标签 | `prefix_causality_audit_invalid` |

## 1. 继承
- D0/D1/D1-R label 逐项核对通过；模型 revision/config/tokenizer 哈希与 D1 一致。
- 仅使用 train 587 + dev 195；final-reserve 197 未读取/评分/缓存/提取。
- D0 中间态 hash 缺陷按 D1-R `provenance_amendment.md` 处理。

## 2. 行为与 hidden-state 等价审计
- dev 195×4 前向（`output_hidden_states=True`）重算评分：predicted_label 780/780 一致；l_A/l_B/d_raw 与 D1 对齐；
- aggregate：OO=1.000, OS=1.000, SO=0.928, SS=0.241，与 D1 完全一致。
- 证明含 hidden states 的前向未改变 Judge 行为。

## 3. token span 定位
- `apply_chat_template(tokenize=False)` → offset-mapping → token ids 与 `tokenize=True` 逐输入一致；
- R_end（Reference 正文末）、C_end（Candidate 正文末）、D_pos（prompt_len-1）全部唯一确定；
- 60 组抽查（train 30 + dev 30）明细见 `token_span_mapping_audit.md`。

## 4. SS 类别容量
- train：y=1=468（≥100），y=0=119（≥100）；dev：y=1=148（≥30），y=0=47（≥30）。
- 容量门通过（`ss_label_capacity_audit.csv`）。

## 5. 主 readout：M_ref_rep
- 每层 R_end hidden state 训练 L2 logistic（class_weight=balanced，C∈[0.0001..1.0]），train 上 Stratified 5-fold group-level CV 选层/C：layer=20, C=0.01，CV AUROC=0.9297；
- dev 冻结评测：**AUROC=0.9027**（95% CI [0.835, 0.957]），AUPRC=0.9562（[0.921, 0.985]），balanced acc=0.8588（`metrics_primary_dev.csv`）；
- B_surface（9 个预注册表面特征）：dev AUROC=0.620, AUPRC=0.821；**ΔAUPRC(M−B_surface) 95% CI [0.060, 0.208]，lower>0**；
- B_decision（−d_raw，仅上界诊断）：AUROC=AUPRC=1.0，**不可作为公平比较或机制解释**（`decision_score_diagnostic.csv`）；
- permutation-null（200 次，冻结 layer/C）：真实 AUROC=0.9027 > null 97.5%=0.608（`permutation_null_audit.csv`）。

## 6. 四格表示定位
- `Δ_ref / Δ_candidate / Δ_interaction` 在 R_end/C_end/D_pos 各层 L2 norm 统计见 `four_cell_representation_contrasts.csv`；
- 该表仅用于 D3 检查顺序的参考，不声称存在单一稳定方向/低维子空间/因果组件。
- train/dev 答案字符串重叠与 swap donor split 关系见 `swap_overlap_disclosure.md`（描述性披露）。

## 7. D1-R 模板迁移诊断（prefix_causality_audit）
- T1 SS acc=0.1692、T2 SS acc=0.1128，均复现 D1-R（0.169/0.113）；
- 冻结 R_end 风险分数在 T1 error 标签上 AUROC=0.934、T2 上 AUROC=0.952；
- **但 prefix_causality_audit 失败**：
  - 要求：T0/T1/T2 的 R_end 数值完全一致；
  - 实测：T2（与 T0 等长）R_end 与 T0 完全一致（max diff=0.0），而 T1（更长 4 tokens）在 27/195 group 上 R_end 与 T0 数值不同（max diff=7.5）；
  - 证据链：前缀 token ids 完全一致；R_end token 位置一致；但完整前向下 R_end 的 hidden state 与 logits 随序列总长度变化（eager 与 sdpa 均复现）；截断到 R_end 前缀后与 T0 完全一致（diff=0.0）；
  - 结论：R_end 数值并非严格的“读到 Reference 即冻结”的纯前缀状态，而对**序列总长度**（含后续 Candidate 文本长度）敏感；受影响的 group 多为多词/较长 reference。
- 因此，即使主 Probe（T0 内）结果很强，**按预注册协议停止**：`prefix_causality_audit_invalid`。
  - 该结论不否定 reference-binding 研究问题本身，也不允许进入 D3；
  - T1/T2 未用于选层、选 C、选特征或修改主结论。

## 8. 最终判定
- 所有继承与技术审计（继承、行为复现、token span、容量）通过；
- 表示层门（M_ref_rep AUROC、CI、ΔAUPRC、T1/T2 迁移、permutation-null）**若单独评估均通过**；
- 但 prefix_causality_audit 触发停止条件 → 最终标签 `prefix_causality_audit_invalid`；
- **不得自动进入 D3。**

## 9. 交付物清单
- final_report.md / inheritance_audit.md / model_access_audit.md / score_hidden_equivalence_audit.md /
- token_span_mapping_audit.md / hidden_state_manifest.json / ss_label_capacity_audit.csv / train_cv_by_layer.csv /
- metrics_primary_dev.csv / metrics_template_transfer_dev.csv / four_cell_representation_contrasts.csv /
- surface_baseline_spec.json / surface_baseline_metrics.csv / decision_score_diagnostic.csv /
- swap_overlap_disclosure.md / permutation_null_audit.csv / failure_examples.md / artifacts/decision.json
