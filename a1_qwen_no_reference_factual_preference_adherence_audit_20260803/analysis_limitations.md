# analysis_limitations.md

## 局限与边界

1. 本轮是**行为关联审计**，不是 hidden-state 机制实验；不证明参数知识为唯一因果原因。
2. `k` 为无 Reference 双顺序 teacher-forced 偏好分数，衡量行为层面的候选偏好，不直接度量参数知识量。
3. AUROC / logistic β 反映的是关联强度与方向，不是因果；可能存在未观测混淆（题目难度、表达差异等）。
4. SciQ（195 group，y_SS=1 占 148）与 PopQA（2,815 group，y_SS=1 占 144）正类比例差异大，
   AUPRC 的绝对水平受 prevalence 影响；解读以 AUROC 与 CI 为主。
5. PopQA 中 `color` 样本量仅 4，其 relation 分层统计不可靠，如实保留、不作单独结论。
6. 不授权进入 intervention，也不改变既有 D1、D2-R1、D3 或 E1 的结论。
