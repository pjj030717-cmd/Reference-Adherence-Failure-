# D4-Q1：Qwen True-Prefix Risk Monitor 的 Final-Reserve 确认

## 问题与结果

| 问题 | 结果 |
|---|---|
| final-reserve 泄露 group 是否已永久隔离？ | 是（1 个隔离，D4-M0 隔离审计工件 + 本轮 quarantine audit） |
| 未泄露 final group 数量是否为 196？ | 是（D0 final split 197 − 1 泄露） |
| D2-R1 的 L18/R_end/Probe 规格是否唯一继承？ | 是（layer 18、C=0.01、R_end offset-mapping、9 特征 B_surface） |
| true-prefix 是否在 Candidate 前严格截断？ | 是（196/196 无 Candidate/Answer:/generation token，60 组重复前向 diff=0） |
| M_rep 的 final AUROC | 0.925853 |
| B_surface 的 final AUROC | 0.682999 |
| ΔAUROC 的 bootstrap CI | [0.138414, 0.351389] |
| permutation-null 是否通过？ | 是（real 0.925853 > null 97.5% 0.589433） |
| 是否支持 H2？ | 是 |
| 最终标签 | **qwen_true_prefix_monitor_final_confirmed** |

## 方法概要

- 目标模型：`Qwen/Qwen2.5-7B-Instruct`（revision `a09a3545…`），BF16、`eval()`、`torch.inference_mode()`、`batch_size=1`。
- A/B 读出：`" A"`→362、`" B"`→425；`d_raw = log p(" A") − log p(" B")`；teacher-forced 位置 `prompt_len−1`；`y=1 iff d_raw<=0`（SS 错拒）。
- 输入：T0 `The answer is <answer>.` 的完整 SS prompt；true prefix = `full_input_ids[:R_end+1]`（R_end = Reference Answer 正文最后一个非空白 token，D2-R1 offset-mapping）。
- M_rep：D2-R1 冻结 Probe（L18 × R_end hidden state，L2 LogReg C=0.01，StandardScaler train-only）。dev 只读复现 AUROC=0.9139（与 D2-R1 记录一致）。
- B_surface：D2-R1 冻结 9 特征表面 baseline（LogReg C=1.0，train-only）。dev 只读复现 AUROC=0.6208（一致）。
- final 评分：仅 196 个允许 group 的一次性 SS A/B 读出；未读取、评分、缓存或写出隔离 group。

## Phase 0 隔离与继承

- `final_reserve_quarantine_audit.md`：D0 final split 197 → 泄露 1（`0075758e…`）→ 允许 196。
- 允许集合索引哈希 `e1c36f65…`，与 D0 final split（`9fe440d6…`）一致性验证通过。
- `allowed_final_group_manifest.json` 仅含 source_group_id / group_hash / split / allow_status，无任何原始文本。

## Phase 0.3 冻结重建

| 模型 | 复现 dev AUROC | D2-R1 记录 | 复现 dev AUPRC | 记录 |
|---|---|---|---|---|
| M_rep | 0.913887 | 0.913887 | 0.963219 | 0.963219 |
| B_surface | 0.620759 | 0.620759 | 0.818192 | 0.818192 |

## Phase 1 回归与机械合同

- A/B readout 回归：24/24（A 12/12、B 12/12）、ties=0、greedy 24/24 一致。
- true-prefix 合同：196/196 R_end 唯一定位；196/196 prefix 无 Candidate span、无 `Answer:`/generation prompt token；60 组重复前向逐层 max_abs_diff=0。

## Phase 2 final 一次性评估

| 指标 | 值 |
|---|---|
| n_total / n_y1 / n_y0 / prevalence | 196 / 157 / 39 / 0.8010 |
| AUROC(M_rep) | 0.925853 |
| AUPRC(M_rep) | 0.976268 |
| AUROC(B_surface) | 0.682999 |
| AUPRC(B_surface) | 0.885240 |
| ΔAUROC(M_rep − B_surface) | +0.242855 |
| ΔAUPRC(M_rep − B_surface) | +0.091028 |
| bootstrap 95% CI of ΔAUROC | [0.138414, 0.351389] |
| permutation-null 97.5% | 0.589433（real 0.925853） |

容量门 PASS（n_y1=157≥30、n_y0=39≥30）。统计设置：group-paired bootstrap 2000（seed 20260812）；permutation 200（seed 20260813），仅置换 y 标签。

## Phase 3 决定门

| 门 | 状态 |
|---|---|
| AUROC(M_rep) ≥ 0.70 | PASS（0.9259） |
| ΔAUROC bootstrap CI lower > 0 | PASS（0.1384） |
| 真实 AUROC > permutation-null 97.5% | PASS（0.9259 vs 0.5894） |
| 无 NaN/inf | PASS |
| 196 组全部唯一预测 | PASS |

最终标签：**qwen_true_prefix_monitor_final_confirmed**

## 结论边界

在本轮允许的范围内，可声称：**在 Qwen2.5-7B 与独立 JAR-style SciQ 设置中，Candidate 出现前的真实 Reference prefix 含有可泛化预测后续 SS false rejection 的表示信号，且优于已冻结的表面 baseline**（final AUROC 0.926 vs 0.683，ΔAUROC CI 不跨 0，permutation 显著）。

不可声称：

- 不构成参数知识是错误唯一原因的证据（未测量知识本身）。
- 不构成跨模型普适机制（仅 Qwen2.5-7B 一个模型）。
- 不构成可直接修复的因果方向（未做 activation intervention / 因果扫描）。
- 不涉及 T1/T2 模板稳健性、其他 cell（OO/OS/SO）、prompt baseline 或 Mistral。

## 交付物清单

| 文件 | 说明 |
|---|---|
| final_report.md | 本报告 |
| final_reserve_quarantine_audit.md | 隔离审计 |
| allowed_final_group_manifest.json | 196 允许 group（仅 hash） |
| inheritance_audit.md | 继承审计 |
| model_access_audit.md | 模型访问审计 |
| frozen_probe_reconstruction_audit.md | Probe 重建审计 |
| frozen_surface_baseline_reconstruction_audit.md | B_surface 重建审计 |
| synthetic_readout_regression.csv | 24 synthetic readout |
| true_prefix_final_contract_audit.csv | 196 组机械合同 |
| final_ss_score_and_label_manifest.csv | 196 组 SS 评分与标签 |
| final_prediction_manifest.csv | 196 组预测 |
| metrics_final.csv | 主指标 |
| bootstrap_final_metrics.csv | bootstrap CI |
| permutation_null_final.csv | permutation null |
| failure_examples.md | 仅 hash + 数值 |
| artifacts/decision.json | 决定 |
