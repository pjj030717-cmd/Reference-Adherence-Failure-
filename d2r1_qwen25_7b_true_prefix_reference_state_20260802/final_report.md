# D2-R1：真实截断 reference-prefix 的表示资格复检

| 问题 | 结果 |
|---|---|
| D0/D1/D1-R/D2 是否被唯一继承？ | 是（D2 label 原样 `prefix_causality_audit_invalid`，未改动） |
| 是否真正截断在 Reference Answer 末尾？ | 是（`prefix_input_ids = full_input_ids[:R_end+1]`，无 Candidate/Answer:/generation-prompt token） |
| T0/T1/T2 的 prefix token ids 是否完全相同？ | 是（dev 195/195；R_end token id 与位置一致；prefix SHA256 一致） |
| SS error train/dev 容量是否足够？ | 是（train 468/119；dev 148/47） |
| 真 prefix Probe 是否优于 B_surface？ | 是（ΔAUPRC CI lower>0） |
| 是否迁移到 T1/T2？ | 是（行为复现 D1-R 0.169/0.113；冻结 Probe AUROC 0.943/0.961） |
| 是否读取/评分 final-reserve？ | 否 |
| 是否运行 intervention？ | 否 |
| 是否允许进入 D3？ | 是 |
| 最终标签 | `true_prefix_reference_state_signal_localized` |

## 1. 继承边界
- D0/D1/D1-R/D2 label 逐项核对；D2 正式标签 `prefix_causality_audit_invalid` 原样保留；
- 模型 revision/config/tokenizer 哈希与 D1 一致；T0 模板 SHA256 一致；
- 仅使用 train 587 + dev 195；final-reserve 197 未读取/评分/缓存/提取；
- 仅继承 D2 的 T0 SS 行为评分表（dev SS 标签逐行核对了 D1 的 `four_cell_scores_dev.csv`），未加载/复用 D2 任何 hidden-state 数组。

## 2. 唯一修改：真实 prefix 截断
- D2 的无效点：完整序列前向的事后读取使 R_end state 随序列总长度漂移；
- D2-R1：`prefix_input_ids = full_input_ids[:R_end+1]`，截断序列单独前向，读最后一位 hidden state；
- 规格见 `true_prefix_input_spec.md`。

## 3. prefix 合同性审计
- dev 全部 195 SS group：T0/T1/T2 在 0..R_end 范围内 token ids 完全相同；R_end token id 与 position 完全相同；prefix SHA256 一致；
- 相同 prefix_input_ids 重复前向所有层 h_prefix 完全一致（max diff = 0.0）；
- 明细见 `true_prefix_contract_audit.csv`（含 train 30 + dev 30 抽样行列）。

## 4. 行为标签与容量
- y_SS_error 定义与 D2/D1 完全一致（T0 全 prompt SS Judge 判决）；dev 标签与 D1 逐行一致；
- train：y=1=468（≥100），y=0=119（≥100）；dev：y=1=148（≥30），y=0=47（≥30）；容量门通过。

## 5. M_true_prefix_rep
- 每层 h_prefix 训练 L2 logistic（class_weight=balanced，C∈[0.0001..1.0]），train Stratified 5-fold group CV 选层/C：layer=18, C=0.01，CV AUROC=0.9195；
- dev 冻结评测：**AUROC=0.9139**（95% CI [0.856, 0.963]），AUPRC=0.9632（[0.933, 0.988]），balanced acc（见 metrics_true_prefix_dev.csv）；
- B_surface：dev AUROC=0.6208, AUPRC=0.8182；**ΔAUPRC(M−B) 95% CI [0.067, 0.219]，lower>0**；
- permutation-null（200 次，冻结 layer/C）：真实 0.9139 > null 97.5%=0.595。

## 6. 模板迁移（冻结诊断）
- T1 SS acc=0.1692、T2 SS acc=0.1128，复现 D1-R（0.169/0.113）；
- 冻结 M_true_prefix_rep 在 T1 error 标签上 AUROC=0.9433、AUPRC=0.9855；T2 上 AUROC=0.9611、AUPRC=0.9950；
- T1/T2 仅用于冻结迁移诊断，未参与选层、选 C、选特征或改主结论。

## 7. 决定门判定
| 条件 | 结果 |
|---|---|
| 1. D0/D1/D1-R/D2 继承有效 | ✓ |
| 2. 真截断 prefix 合同审计通过 | ✓ |
| 3. SS train/dev 容量通过 | ✓ |
| 4. M_true_prefix_rep dev AUROC >= 0.65 | ✓（0.9139） |
| 5. dev AUROC CI lower > 0.55 | ✓（0.856） |
| 6. ΔAUPRC vs B_surface CI lower > 0 | ✓（0.067） |
| 7. T1/T2 冻结 AUROC 均 >= 0.60 | ✓（0.943/0.961） |
| 8. permutation-null 97.5 分位 < 真实 | ✓（0.595 < 0.914） |
| 9. final-reserve 未读取 | ✓ |

**全部通过 → 允许进入 D3。**

## 8. 与 D2 的关系（边界说明）
- D2 正式标签 `prefix_causality_audit_invalid` 原样保留（见 `d2_invalid_result_boundary_note.md`）；
- 真截断修正后，reference-prefix 阶段的线性 readout 信号成立且强于 D2 的完整序列版本；
- 这属于单变量修正后的资格复检，不否定 D2 关于"完整序列 R_end 数值随长度漂移"的审计发现。

## 9. 交付物清单
- final_report.md / inheritance_audit.md / true_prefix_input_spec.md / true_prefix_contract_audit.csv /
- prefix_hidden_state_manifest.json / ss_label_capacity_audit.csv / train_cv_by_layer.csv / metrics_true_prefix_dev.csv /
- surface_baseline_metrics.csv / metrics_template_transfer_dev.csv / permutation_null_audit.csv /
- d2_invalid_result_boundary_note.md / failure_examples.md / artifacts/decision.json
