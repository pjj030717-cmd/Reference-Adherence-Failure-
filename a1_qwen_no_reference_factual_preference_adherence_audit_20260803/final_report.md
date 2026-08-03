# final_report.md

## 摘要

| 问题 | 结果 |
|---|---|
| 是否只使用 SciQ / PopQA development group？ | 是（SciQ dev 195 / PopQA dev 2,815；与既有 dev manifest 逐组一致） |
| 是否零 final-reserve / train 文本模型接触？ | 是（未读取、未评分、未缓存任何 train / final-reserve 文本） |
| 无 Reference A/B 读出语义是否有效？ | 有效（合成 24 对：overall 24/24，A 12/12，B 12/12，ties=0，greedy 一致 24/24） |
| SciQ 中 `k` 是否预测 SS false rejection？ | 是（AUROC 0.732，95% CI [0.644, 0.811]） |
| PopQA 中 `k` 是否预测 SS false rejection？ | 是（AUROC 0.879，95% CI [0.846, 0.910]） |
| PopQA relation 控制后关联是否保留？ | 保留（relation-FE β1(z(k))=2.043，95% CI [1.486, 2.899]） |
| 是否支持“无 Reference 事实偏好与 SS 错拒有关”的行为解释？ | 是（跨数据集、控制长度与 PopQA relation 后仍保留） |
| 最终标签 | **no_reference_factual_preference_association_supported** |

## 结论

模型在无 Reference 条件下对原始答案的事实偏好，与其后续拒绝 swapped-Reference 一致候选的行为存在跨数据集、控制长度与 PopQA relation 后仍保留的正向关联。

该结论与该解释一致，但不足以证明参数知识是唯一原因。

## Phase 1：事实偏好分数汇总

| 数据集 | n | d_1 mean | d_1 med | d_2 mean | d_2 med | k mean | k med | order-consistent | tie | k>0 | k<0 | k=0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SciQ | 195 | 20.448 | 21.781 | 20.598 | 21.594 | 20.523 | 21.688 | 0.995 | 0.0000 | 0.990 | 0.010 | 0.0000 |
| PopQA | 2815 | 7.999 | 9.500 | 6.557 | 8.438 | 7.278 | 8.688 | 0.836 | 0.0032 | 0.749 | 0.250 | 0.0011 |

## Phase 2：主关联分析

### 2A 数据集内排序

| 数据集 | AUROC | 95% CI | AUPRC | 95% CI | y_SS=0 med k | y_SS=1 med k | Cliff's δ | MWU p |
|---|---|---|---|---|---|---|---|---|
| SciQ | 0.732 | [0.644, 0.811] | 0.874 | [0.813, 0.931] | 20.012 | 22.047 | -0.463 | 1.76e-06 |
| PopQA | 0.879 | [0.846, 0.910] | 0.395 | [0.317, 0.488] | 8.047 | 21.074 | -0.759 | 3.33e-53 |

> Cliff's δ 按 `cliffs_delta(k_{ySS=0}, k_{ySS=1})` 计算：负值表示 y_SS=1 组（SS 错拒组）的 k 更高，
> 与"k 越高越易错拒"方向一致。MWU p 为描述性双尾 p 值。

### 2B 长度控制 logistic（β 为 z-score 后系数）

| 数据集 | β1 z(k) | OR=exp(β1) | 95% CI (β1) | β2 z(q_len) | β3 z(len r_o) | β4 z(len r_s) |
|---|---|---|---|---|---|---|
| SciQ | 0.763 | 2.145 | [0.378, 1.497] | 0.132 | 0.012 | -0.055 |
| PopQA | 2.713 | 15.076 | [2.086, 3.660] | -0.298 | -0.165 | -0.330 |

### 2C PopQA relation fixed-effects

| 模型 | n | β1 z(k) | OR | 95% CI (β1) | reference relation |
|---|---|---|---|---|---|
| relation FE | 2815 | 2.043 | 7.717 | [1.486, 2.899] | author |

relation 分层描述表见 `popqa_relation_descriptive_audit.csv`；仅当 relation 同时含正负类且 n>=30 时报告 AUROC。

## 预注册规则核查

```text
sciq_auroc_ge_065:               OK (0.732 >= 0.65)
sciq_auroc_ci_lower_gt_050:      OK (0.644 > 0.50)
sciq_lenctrl_b1_ci_lower_gt_0:   OK (0.378 > 0)
popqa_relfe_b1_ci_lower_gt_0:    OK (1.486 > 0)
both_ds_ySS1_median_gt_ySS0:     OK (SciQ 22.05>20.01, PopQA 21.07>8.05)
```

## 结语

本轮是行为关联审计，不是 hidden-state 机制实验；
不证明参数知识为唯一因果原因；
不授权进入 intervention，也不改变既有 D1、D2-R1、D3 或 E1 的结论。
