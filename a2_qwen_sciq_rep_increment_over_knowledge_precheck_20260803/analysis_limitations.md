# analysis_limitations.md

## 局限与边界

1. 本轮仅比较 SciQ development（195 group）上的增量预测价值；它不构成新的 final confirmation。
2. `ρ` 来自 D4-Q1 冻结 Probe（train-only 重建），`k` 按 A1 规范重算；二者都只是预测分数，
   不证明 ρ 是参数知识冲突的因果中介。
3. M_rep / M_hybrid 的 AUROC 点估计在 dev 上较高，但 dev 样本仅 195、正类 148；bootstrap CI 已给出区间。
4. M_hybrid 仅在 train 上拟合一次，无 CV/网格搜索/阈值优化；dev 仅评分一次。
5. Recall@10% 定义：按分数取最高 10% 覆盖（topk=20），计算其中 y=1 的召回。
6. 本轮不授权 activation intervention；若需论文级独立比较，须在未接触的 H1 合格模型 × 数据 setting 上预注册确认。
