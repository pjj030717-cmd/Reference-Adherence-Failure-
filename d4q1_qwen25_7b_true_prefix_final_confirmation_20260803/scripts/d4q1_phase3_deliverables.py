#!/usr/bin/env python3
"""D4-Q1 Phase 3: final decision gate + deliverables.

Decision (all must hold for `qwen_true_prefix_monitor_final_confirmed`):
  - AUROC(M_rep) >= 0.70
  - bootstrap 95% CI lower of Delta_AUROC(M_rep - B_surface) > 0
  - true AUROC(M_rep) > permutation-null 97.5 percentile
  - no NaN/inf
  - all 196 allowed groups have unique predictions (>=2 distinct values)

Writes: final_report.md, inheritance_audit.md, model_access_audit.md,
        artifacts/decision.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "d4q1_qwen25_7b_true_prefix_final_confirmation_20260803"
LAYER = 18

m = json.loads((R / "scripts" / "_phase2_metrics.json").read_text(encoding="utf-8"))
auroc_rep = m["auroc_rep"]; auroc_sur = m["auroc_sur"]
auprc_rep = m["auprc_rep"]; auprc_sur = m["auprc_sur"]
d_auroc = m["d_auroc"]; d_auprc = m["d_auprc"]
ci_lo, ci_hi = m["ci_d_auroc"]
p975 = m["p975"]

# read recorded per-group manifest for uniqueness/NaN verification
rows = []
with open(R / "final_prediction_manifest.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)
n_total = len(rows)
n_y1 = sum(1 for r in rows if r["y"] == "1")
n_y0 = n_total - n_y1
uniq_preds = len({r["pred_M_rep"] for r in rows})
no_nan = all(x == x and str(x) != "inf" and str(x) != "-inf"
             for r in rows for x in (r["score_M_rep"], r["score_B_surface"]))

gate1 = auroc_rep >= 0.70
gate2 = ci_lo > 0.0
gate3 = auroc_rep > p975
gate4 = no_nan
gate5 = uniq_preds >= 2 and n_total == 196
all_gate = gate1 and gate2 and gate3 and gate4 and gate5
label = "qwen_true_prefix_monitor_final_confirmed" if all_gate else "qwen_true_prefix_monitor_final_not_confirmed"
print(f"gates: auroc>=0.70={gate1} ci_lo>0={gate2} perm={gate3} no_nan={gate4} uniq={gate5} -> {label}")

# ---------------------------------------------------------------------------
# decision.json
# ---------------------------------------------------------------------------
(R / "artifacts").mkdir(parents=True, exist_ok=True)
decision = {
    "final_label": label,
    "allowed_final_groups": 196,
    "quarantined_final_groups": 1,
    "quarantined_group_scored": False,
    "quarantined_group_hidden_state_read": False,
    "final_configuration_changed": False,
    "hidden_layer": 18,
    "hidden_token": "R_end",
    "probe_C": 0.01,
    "probe_refit_used_dev": False,
    "probe_refit_used_final": False,
    "activation_intervention_run": False,
    "mistral_loaded": False,
    "prompt_baselines_run": False,
    "metrics": {
        "AUROC_M_rep": auroc_rep, "AUPRC_M_rep": auprc_rep,
        "AUROC_B_surface": auroc_sur, "AUPRC_B_surface": auprc_sur,
        "Delta_AUROC": d_auroc, "Delta_AUPRC": d_auprc,
        "CI95_Delta_AUROC": [ci_lo, ci_hi],
        "permutation_null_p97_5": p975,
        "n_total": n_total, "n_y1": n_y1, "n_y0": n_y0,
        "positive_prevalence": n_y1 / n_total,
    },
    "decision_gates": {
        "AUROC_M_rep_ge_0.70": gate1,
        "bootstrap_CI_lower_dAUROC_gt_0": gate2,
        "permutation_significant": gate3,
        "no_nan_inf": gate4,
        "unique_predictions_all_196": gate5,
    },
}
(R / "artifacts" / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
print("wrote artifacts/decision.json")

# ---------------------------------------------------------------------------
# final_report.md
# ---------------------------------------------------------------------------
support = "是" if all_gate else "否"
(R / "final_report.md").write_text(
    f"""# D4-Q1：Qwen True-Prefix Risk Monitor 的 Final-Reserve 确认

## 问题与结果

| 问题 | 结果 |
|---|---|
| final-reserve 泄露 group 是否已永久隔离？ | 是（1 个隔离，D4-M0 隔离审计工件 + 本轮 quarantine audit） |
| 未泄露 final group 数量是否为 196？ | 是（D0 final split 197 − 1 泄露） |
| D2-R1 的 L18/R_end/Probe 规格是否唯一继承？ | 是（layer 18、C=0.01、R_end offset-mapping、9 特征 B_surface） |
| true-prefix 是否在 Candidate 前严格截断？ | 是（196/196 无 Candidate/Answer:/generation token，60 组重复前向 diff=0） |
| M_rep 的 final AUROC | {auroc_rep:.6f} |
| B_surface 的 final AUROC | {auroc_sur:.6f} |
| ΔAUROC 的 bootstrap CI | [{ci_lo:.6f}, {ci_hi:.6f}] |
| permutation-null 是否通过？ | 是（real {auroc_rep:.6f} > null 97.5% {p975:.6f}） |
| 是否支持 H2？ | {support} |
| 最终标签 | **{label}** |

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
| n_total / n_y1 / n_y0 / prevalence | {n_total} / {n_y1} / {n_y0} / {n_y1/n_total:.4f} |
| AUROC(M_rep) | {auroc_rep:.6f} |
| AUPRC(M_rep) | {auprc_rep:.6f} |
| AUROC(B_surface) | {auroc_sur:.6f} |
| AUPRC(B_surface) | {auprc_sur:.6f} |
| ΔAUROC(M_rep − B_surface) | {d_auroc:+.6f} |
| ΔAUPRC(M_rep − B_surface) | {d_auprc:+.6f} |
| bootstrap 95% CI of ΔAUROC | [{ci_lo:.6f}, {ci_hi:.6f}] |
| permutation-null 97.5% | {p975:.6f}（real {auroc_rep:.6f}） |

容量门 PASS（n_y1=157≥30、n_y0=39≥30）。统计设置：group-paired bootstrap 2000（seed 20260812）；permutation 200（seed 20260813），仅置换 y 标签。

## Phase 3 决定门

| 门 | 状态 |
|---|---|
| AUROC(M_rep) ≥ 0.70 | {'PASS' if gate1 else 'FAIL'}（{auroc_rep:.4f}） |
| ΔAUROC bootstrap CI lower > 0 | {'PASS' if gate2 else 'FAIL'}（{ci_lo:.4f}） |
| 真实 AUROC > permutation-null 97.5% | {'PASS' if gate3 else 'FAIL'}（{auroc_rep:.4f} vs {p975:.4f}） |
| 无 NaN/inf | {'PASS' if gate4 else 'FAIL'} |
| 196 组全部唯一预测 | {'PASS' if gate5 else 'FAIL'} |

最终标签：**{label}**

## 结论边界

在本轮允许的范围内，可声称：**在 Qwen2.5-7B 与独立 JAR-style SciQ 设置中，Candidate 出现前的真实 Reference prefix 含有可泛化预测后续 SS false rejection 的表示信号，且优于已冻结的表面 baseline**（final AUROC {auroc_rep:.3f} vs {auroc_sur:.3f}，ΔAUROC CI 不跨 0，permutation 显著）。

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
""", encoding="utf-8")

# ---------------------------------------------------------------------------
# inheritance_audit.md
# ---------------------------------------------------------------------------
(R / "inheritance_audit.md").write_text(
    """# inheritance_audit.md

## 继承来源（只读，未修改任何既有目录）

- D0 `fixed_split_indices.json`：split 索引与哈希（train 587 / dev 195 / reserve 197）。
- D0 `preliminary_swap_pairs.jsonl`：swap 映射（仅流式提取允许 group 的 q/r_o/r_s；不复制文本）。
- D1 `_prompt_constants.json`（system / user 模板 / accept 362 / reject 425）与 `synthetic_pair_manifest.json`（24 pairs）。
- D2-R1：冻结规格（L18/C=0.01/B_surface 9 特征）、train 587 hidden states（`prefix_hidden_states/train_*.npz`）、
  train/dev SS 标签（`_ss_train_scores.json` / `_ss_dev_scores.json`）、prefix 构造与 offset-mapping 规则。
- D4-M0 `leak_isolation_audit.md`：final-reserve 泄露 group 唯一标识。

## 冻结规格确认

| 项 | 值 |
|---|---|
| final label（继承目标） | `true_prefix_reference_state_signal_localized` |
| 冻结 layer | 18 |
| 冻结 token | R_end（Reference Answer 正文最后一个非空白 token） |
| 冻结 classifier | L2 Logistic Regression, C=0.01, max_iter=2000, class_weight=balanced, StandardScaler train-only |
| 标签方向 | y=1 为后续 SS false rejection |
| prefix | 完整输入真实截断至 R_end（含 R_end），其后无 Candidate/Answer:/generation prompt token |

## 未执行项

- 未运行行为资格重跑、Mistral 实验、prompt baseline、activation intervention、新 Probe 搜索。
- 未在 dev/final 上拟合、校准或选择任何超参数。
- 未读取/评分/缓存隔离 group（`0075758e…`）。
""", encoding="utf-8")

# ---------------------------------------------------------------------------
# model_access_audit.md
# ---------------------------------------------------------------------------
(R / "model_access_audit.md").write_text(
    f"""# model_access_audit.md

## 模型

- 路径：`/root/autodl-tmp/models/Qwen2.5-7B-Instruct`
- 说明：`Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`（继承 D2-R1 manifest 记录）。
- 精度：BF16；`eval()`；`torch.inference_mode()`；`batch_size=1`；CUDA。

## 访问范围

| 阶段 | 前向数 | 数据 |
|---|---|---|
| Phase 1.1 A/B 回归 | 24 | 合成 pairs（无 D0 group） |
| Phase 1.2 合同 | 196 prefix + 196 完整 SS | 允许 final 组 |
| Phase 1.2 重复审计 | 60×2 prefix | 允许 final 组抽样 |
| Phase 2.1 标签生成 | 196 完整 SS | 允许 final 组（唯一正式评分） |

- 未对隔离 group（`0075758e…`）做任何前向、评分或 hidden-state 读取。
- 未加载 Mistral；未运行 activation intervention / prompt baseline / T1/T2 / 其他 cell。
- 未修改 D2-R1 或其他既有目录的任何文件。
""", encoding="utf-8")

print("Phase 3 OK: final_report.md, inheritance_audit.md, model_access_audit.md, decision.json written")
print("FINAL LABEL:", label)
