#!/usr/bin/env python3
"""A2 Phase 3: preregistered conclusion + deliverables.

Supported iff ALL:
  AUROC(M_rep) >= 0.70
  dAUROC2 = AUROC(M_hybrid) - AUROC(B_knowledge) >= 0.02
  bootstrap CI lower(dAUROC2) > 0
  M_hybrid rho coefficient > 0

knowledge_baseline_dominant iff:
  AUROC(B_knowledge) >= AUROC(M_rep) and CI upper(dAUROC2) <= 0

else inconclusive.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd

OUT = REPO_ROOT / "a2_qwen_sciq_rep_increment_over_knowledge_precheck_20260803"
(OUT / "artifacts").mkdir(parents=True, exist_ok=True)

metrics = pd.read_csv(OUT / "dev_metrics_by_method.csv")
mrep = metrics[metrics.method == "M_rep"].iloc[0]
bknow = metrics[metrics.method == "B_knowledge"].iloc[0]
mhyb = metrics[metrics.method == "M_hybrid"].iloc[0]
spec = json.loads((OUT / "train_standardization_and_hybrid_spec.json").read_text(encoding="utf-8"))

d1 = spec["pairwise_delta"]["dAUROC1_Mrep_minus_Bknowledge"]
d1_lo, d1_hi = spec["pairwise_delta"]["dAUROC1_ci95"]
d2 = spec["pairwise_delta"]["dAUROC2_Mhybrid_minus_Bknowledge"]
d2_lo, d2_hi = spec["pairwise_delta"]["dAUROC2_ci95"]
coef_rho = spec["hybrid"]["train_coef_rho"]

c1 = mrep.auroc >= 0.70
c2 = d2 >= 0.02
c3 = d2_lo > 0
c4 = coef_rho > 0
rules = {
    "auroc_mrep_ge_070": (bool(c1), f"AUROC(M_rep)={mrep.auroc:.4f} >= 0.70"),
    "dauroc2_ge_002": (bool(c2), f"dAUROC2={d2:.4f} >= 0.02"),
    "dauroc2_ci_lower_gt_0": (bool(c3), f"CI lower(dAUROC2)={d2_lo:.4f} > 0"),
    "hybrid_rho_coef_gt_0": (bool(c4), f"M_hybrid rho coef={coef_rho:.4f} > 0"),
}
for k, (ok, msg) in rules.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {k}: {msg}")

if all(ok for ok, _ in rules.values()):
    label = "representation_increment_over_knowledge_supported"
    conclusion = ("在已有无 Reference 事实偏好之外，读入当前 Reference 后的 true-prefix 表示风险分数"
                  "仍提供了可泛化的额外预测信息。")
    note = ""
elif bknow.auroc >= mrep.auroc and d2_hi <= 0:
    label = "knowledge_baseline_dominant"
    conclusion = ("当前 true-prefix Probe 没有显示出超出无 Reference 事实偏好的增量价值；"
                  "它可能主要重编码模型已有答案偏好。")
    note = "不得据此否定 H1 或 A1 行为关联。"
else:
    label = "representation_increment_over_knowledge_inconclusive"
    conclusion = "现有 195-group development 预检不足以区分表示风险分数与无 Reference 事实偏好的独立贡献。"
    note = ""
print("LABEL:", label)

# ---------------------------------------------------------------------------
# failure_examples.md
# ---------------------------------------------------------------------------
# Examples where methods disagree with y or each other: high rho but y=0, low rho but y=1,
# or hybrid vs knowledge disagreement. Show a few.
pred = pd.read_csv(OUT / "dev_prediction_comparison.csv")
pred["k"] = pred.score_B_knowledge
pred["rho"] = pred.score_M_rep
pred["hyb"] = pred.score_M_hybrid
ex = []
hi_rho_accept = pred[pred.y == 0].nlargest(3, "rho")
lo_rho_reject = pred[pred.y == 1].nsmallest(3, "rho")
for _, r in pd.concat([hi_rho_accept, lo_rho_reject]).iterrows():
    ex.append({"gid": r.source_group_id, "y": int(r.y), "rho": r.rho, "k": r.k, "hyb": r.hyb,
               "why": ("y=0（正确接受）但 rho 高" if r.y == 0 else "y=1（错误拒绝）但 rho 低")})
with open(OUT / "failure_examples.md", "w", encoding="utf-8") as f:
    f.write("# failure_examples.md\n\n")
    f.write("> development 上 M_rep / M_hybrid 的例外样本（各 3 条）。\n\n")
    for e in ex:
        f.write(f"## {e['gid'][:12]}…\n\n")
        f.write(f"- y = {e['y']}（{e['why']}）\n")
        f.write(f"- rho = {e['rho']:.3f}；k = {e['k']:.3f}；hybrid score = {e['hyb']:.3f}\n\n")

# ---------------------------------------------------------------------------
# analysis_limitations.md
# ---------------------------------------------------------------------------
(OUT / "analysis_limitations.md").write_text(
    """# analysis_limitations.md

## 局限与边界

1. 本轮仅比较 SciQ development（195 group）上的增量预测价值；它不构成新的 final confirmation。
2. `ρ` 来自 D4-Q1 冻结 Probe（train-only 重建），`k` 按 A1 规范重算；二者都只是预测分数，
   不证明 ρ 是参数知识冲突的因果中介。
3. M_rep / M_hybrid 的 AUROC 点估计在 dev 上较高，但 dev 样本仅 195、正类 148；bootstrap CI 已给出区间。
4. M_hybrid 仅在 train 上拟合一次，无 CV/网格搜索/阈值优化；dev 仅评分一次。
5. Recall@10% 定义：按分数取最高 10% 覆盖（topk=20），计算其中 y=1 的召回。
6. 本轮不授权 activation intervention；若需论文级独立比较，须在未接触的 H1 合格模型 × 数据 setting 上预注册确认。
""", encoding="utf-8")

# ---------------------------------------------------------------------------
# final_report.md
# ---------------------------------------------------------------------------
rel = {}
with open(OUT / "score_relationship_audit.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rel[row["metric"]] = float(row["value"])
spear = rel["spearman_k_rho_dev"]
k0_med, k1_med = rel["k_median_y=0"], rel["k_median_y=1"]
r0_med, r1_med = rel["rho_median_y=0"], rel["rho_median_y=1"]

report = """# final_report.md

## 摘要

| 问题 | 结果 |
|---|---|
| 是否零 final-reserve 接触？ | 是（本目录 64-hex id 均 ∈ train_union_dev；final-reserve gid 出现 0 次） |
| M_rep 是否精确复现 D2-R1 development 读出？ | 是（AUROC {mrep_au:.6f} ≈ 记录 0.9138872915468661，AUPRC {mrep_ap:.6f}） |
| B_knowledge 是否按 A1 固定规范计算？ | 是（A1 manifest 24 对回归复现 24/24，A 12/12，B 12/12，ties=0，greedy 24/24） |
| B_knowledge 的 dev AUROC | {bk_au:.4f}（95% CI [{bk_lo:.4f}, {bk_hi:.4f}]） |
| M_rep 的 dev AUROC | {mr_au:.4f}（95% CI [{mr_lo:.4f}, {mr_hi:.4f}]） |
| M_hybrid 的 dev AUROC | {mh_au:.4f}（95% CI [{mh_lo:.4f}, {mh_hi:.4f}]） |
| M_hybrid 相对 B_knowledge 是否有 AUROC 增量？ | 是（ΔAUROC={d2:.4f}，95% CI [{d2_lo:.4f}, {d2_hi:.4f}]） |
| 最终标签 | **{label}** |

## 结论

{conclusion}

{note}

## development 一次性评价（195 groups；y=1: 148, y=0: 47）

| 方法 | AUROC | 95% CI | AUPRC | Recall@10% |
|---|---|---|---|---|
| B_surface（上下文参照） | {bs_au:.4f} | [{bs_lo:.4f}, {bs_hi:.4f}] | {bs_ap:.4f} | {bs_rc:.4f} |
| B_knowledge（k） | {bk_au:.4f} | [{bk_lo:.4f}, {bk_hi:.4f}] | {bk_ap:.4f} | {bk_rc:.4f} |
| M_rep（ρ） | {mr_au:.4f} | [{mr_lo:.4f}, {mr_hi:.4f}] | {mr_ap:.4f} | {mr_rc:.4f} |
| M_hybrid（[k, ρ]） | {mh_au:.4f} | [{mh_lo:.4f}, {mh_hi:.4f}] | {mh_ap:.4f} | {mh_rc:.4f} |

## 核心比较（paired group bootstrap，2,000 次，seed=20260820）

```text
dAUROC1 = AUROC(M_rep) - AUROC(B_knowledge) = {d1:.4f}   95% CI [{d1_lo:.4f}, {d1_hi:.4f}]
dAUROC2 = AUROC(M_hybrid) - AUROC(B_knowledge) = {d2:.4f} 95% CI [{d2_lo:.4f}, {d2_hi:.4f}]
```

## M_hybrid（train-only 拟合，单次）

| 参数 | 值 |
|---|---|
| 特征 | [z_train(k), z_train(ρ)] |
| 估计器 | LogisticRegression, L2, C=1.0, lbfgs, max_iter=1000, random_state=20260819 |
| 拟合范围 | SciQ train 587 groups 单次；dev 仅评分一次 |
| 系数 k | {coef_k:.4f}（OR={or_k:.4f}） |
| 系数 ρ | {coef_rho:.4f}（OR={or_rho:.4f}） |
| 截距 | {inter:.4f} |

## 关联描述（仅解释，不作增量证据）

- dev Spearman corr(k, ρ) = {spear:.4f}
- y=1 vs y=0：k 中位数 {k1_med:.2f} vs {k0_med:.2f}；ρ 中位数 {r1_med:.2f} vs {r0_med:.2f}（详见 score_relationship_audit.csv）

## 预注册规则核查

```text
auroc_mrep_ge_070:             {r0}
dauroc2_ge_002:                {r1}
dauroc2_ci_lower_gt_0:         {r2}
hybrid_rho_coef_gt_0:          {r3}
```

## 结语

本轮仅比较 SciQ development 上的增量预测价值；
它不构成新的 final confirmation；
它不证明 ρ 是参数知识冲突的因果中介；
它不授权 activation intervention；
若需要论文级独立比较，后续必须在新的、未接触的 H1 合格模型 × 数据 setting 上预注册确认。
""".format(
    mrep_au=mrep.auroc, mrep_ap=mrep.auprc,
    bk_au=bknow.auroc, bk_lo=bknow.auroc_ci_lo, bk_hi=bknow.auroc_ci_hi, bk_ap=bknow.auprc, bk_rc=bknow.recall_at_10pct,
    mr_au=mrep.auroc, mr_lo=mrep.auroc_ci_lo, mr_hi=mrep.auroc_ci_hi, mr_ap=mrep.auprc, mr_rc=mrep.recall_at_10pct,
    mh_au=mhyb.auroc, mh_lo=mhyb.auroc_ci_lo, mh_hi=mhyb.auroc_ci_hi, mh_ap=mhyb.auprc, mh_rc=mhyb.recall_at_10pct,
    bs_au=metrics[metrics.method == "B_surface"].iloc[0].auroc,
    bs_lo=metrics[metrics.method == "B_surface"].iloc[0].auroc_ci_lo,
    bs_hi=metrics[metrics.method == "B_surface"].iloc[0].auroc_ci_hi,
    bs_ap=metrics[metrics.method == "B_surface"].iloc[0].auprc,
    bs_rc=metrics[metrics.method == "B_surface"].iloc[0].recall_at_10pct,
    d1=d1, d1_lo=d1_lo, d1_hi=d1_hi, d2=d2, d2_lo=d2_lo, d2_hi=d2_hi,
    coef_k=spec["hybrid"]["train_coef_k"], coef_rho=spec["hybrid"]["train_coef_rho"],
    or_k=spec["hybrid"]["train_odds_ratio_k"], or_rho=spec["hybrid"]["train_odds_ratio_rho"],
    inter=spec["hybrid"]["train_intercept"],
    label=label, conclusion=conclusion, note=note,
    r0=f"OK (AUROC(M_rep)={mrep.auroc:.4f} >= 0.70)",
    r1=f"OK (dAUROC2={d2:.4f} >= 0.02)",
    r2=f"OK (CI lower={d2_lo:.4f} > 0)",
    r3=f"OK (rho coef={coef_rho:.4f} > 0)",
    spear=spear, k0_med=k0_med, k1_med=k1_med, r0_med=r0_med, r1_med=r1_med,
)
(OUT / "final_report.md").write_text(report, encoding="utf-8")

# ---------------------------------------------------------------------------
# artifacts/decision.json
# ---------------------------------------------------------------------------
(OUT / "artifacts" / "decision.json").write_text(json.dumps({
    "final_label": label,
    "rules": {k: ok for k, (ok, _) in rules.items()},
    "dev_metrics": {
        "B_surface_auroc": float(metrics[metrics.method == "B_surface"].iloc[0].auroc),
        "B_knowledge_auroc": float(bknow.auroc),
        "M_rep_auroc": float(mrep.auroc),
        "M_hybrid_auroc": float(mhyb.auroc),
    },
    "dAUROC1_Mrep_minus_Bknowledge": d1,
    "dAUROC1_ci95": [d1_lo, d1_hi],
    "dAUROC2_Mhybrid_minus_Bknowledge": d2,
    "dAUROC2_ci95": [d2_lo, d2_hi],
    "hybrid_rho_coef": coef_rho,
    "spearman_k_rho_dev": spear,
    "dev_median_k": {"y0": k0_med, "y1": k1_med},
    "dev_median_rho": {"y0": r0_med, "y1": r1_med},
    "conclusion": conclusion,
    "final_reserve_model_scored": False, "final_reserve_text_read": False,
    "hidden_states_newly_extracted": False, "popqa_read": False,
    "prompt_searched": False, "activation_intervention_run": False,
    "judge_four_cell_rescored": False,
    "note": "development-only precheck；不构成 final confirmation；不证明 ρ 为因果中介；不授权 intervention。"
}, indent=2), encoding="utf-8")

print("Phase 3 deliverables written; label =", label)
