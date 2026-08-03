# final_report.md

## 摘要

| 问题 | 结果 |
|---|---|
| 是否零 final-reserve 接触？ | 是（本目录 64-hex id 均 ∈ train_union_dev；final-reserve gid 出现 0 次） |
| M_rep 是否精确复现 D2-R1 development 读出？ | 是（AUROC 0.913887 ≈ 记录 0.9138872915468661，AUPRC 0.963219） |
| B_knowledge 是否按 A1 固定规范计算？ | 是（A1 manifest 24 对回归复现 24/24，A 12/12，B 12/12，ties=0，greedy 24/24） |
| B_knowledge 的 dev AUROC | 0.7317（95% CI [0.6412, 0.8146]） |
| M_rep 的 dev AUROC | 0.9139（95% CI [0.8551, 0.9607]） |
| M_hybrid 的 dev AUROC | 0.9202（95% CI [0.8652, 0.9646]） |
| M_hybrid 相对 B_knowledge 是否有 AUROC 增量？ | 是（ΔAUROC=0.1885，95% CI [0.0855, 0.2879]） |
| 最终标签 | **representation_increment_over_knowledge_supported** |

## 结论

在已有无 Reference 事实偏好之外，读入当前 Reference 后的 true-prefix 表示风险分数仍提供了可泛化的额外预测信息。



## development 一次性评价（195 groups；y=1: 148, y=0: 47）

| 方法 | AUROC | 95% CI | AUPRC | Recall@10% |
|---|---|---|---|---|
| B_surface（上下文参照） | 0.6208 | [0.5309, 0.7151] | 0.8182 | 0.1149 |
| B_knowledge（k） | 0.7317 | [0.6412, 0.8146] | 0.8745 | 0.1149 |
| M_rep（ρ） | 0.9139 | [0.8551, 0.9607] | 0.9632 | 0.1351 |
| M_hybrid（[k, ρ]） | 0.9202 | [0.8652, 0.9646] | 0.9668 | 0.1351 |

## 核心比较（paired group bootstrap，2,000 次，seed=20260820）

```text
dAUROC1 = AUROC(M_rep) - AUROC(B_knowledge) = 0.1822   95% CI [0.0733, 0.2841]
dAUROC2 = AUROC(M_hybrid) - AUROC(B_knowledge) = 0.1885 95% CI [0.0855, 0.2879]
```

## M_hybrid（train-only 拟合，单次）

| 参数 | 值 |
|---|---|
| 特征 | [z_train(k), z_train(ρ)] |
| 估计器 | LogisticRegression, L2, C=1.0, lbfgs, max_iter=1000, random_state=20260819 |
| 拟合范围 | SciQ train 587 groups 单次；dev 仅评分一次 |
| 系数 k | 0.2433（OR=1.2754） |
| 系数 ρ | 4.9132（OR=136.0746） |
| 截距 | 4.5947 |

## 关联描述（仅解释，不作增量证据）

- dev Spearman corr(k, ρ) = 0.2379
- y=1 vs y=0：k 中位数 22.05 vs 20.01；ρ 中位数 4.31 vs -2.10（详见 score_relationship_audit.csv）

## 预注册规则核查

```text
auroc_mrep_ge_070:             OK (AUROC(M_rep)=0.9139 >= 0.70)
dauroc2_ge_002:                OK (dAUROC2=0.1885 >= 0.02)
dauroc2_ci_lower_gt_0:         OK (CI lower=0.0855 > 0)
hybrid_rho_coef_gt_0:          OK (rho coef=4.9132 > 0)
```

## 结语

本轮仅比较 SciQ development 上的增量预测价值；
它不构成新的 final confirmation；
它不证明 ρ 是参数知识冲突的因果中介；
它不授权 activation intervention；
若需要论文级独立比较，后续必须在新的、未接触的 H1 合格模型 × 数据 setting 上预注册确认。
