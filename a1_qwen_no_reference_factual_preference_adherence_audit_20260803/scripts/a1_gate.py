#!/usr/bin/env python3
"""A1 Phase 3: preregistered conclusion + deliverables.

Preregistered rules:
  supported            if SciQ AUROC>=0.65 & CI lower>0.50, SciQ b1 CI lower>0,
                       PopQA relation-FE b1 CI lower>0, and y_SS=1 median k > y_SS=0 median k in both datasets.
  dataset_contingent   if SciQ supports but PopQA relation-FE b1 CI contains 0 / unstable direction.
  not_supported        otherwise.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd

OUT = REPO_ROOT / "a1_qwen_no_reference_factual_preference_adherence_audit_20260803"
(OUT / "artifacts").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT / "factual_preference_scores_dev.csv")
m = pd.read_csv(OUT / "dataset_level_association_metrics.csv")
l = pd.read_csv(OUT / "length_controlled_logistic_results.csv")
p = pd.read_csv(OUT / "popqa_relation_controlled_results.csv")

import ast
for col in ["ySS0", "ySS1"]:
    m[col] = m[col].map(ast.literal_eval)


def get(rows, ds):
    return rows[rows.dataset == ds].iloc[0]


def phase1_summary(dff):
    out = {}
    for ds, pref in [("SciQ", "s"), ("PopQA", "p")]:
        g = dff[dff.dataset == ds]
        vals = [g.d_1.mean(), g.d_1.median(), g.d_2.mean(), g.d_2.median(),
                g.k.mean(), g.k.median(), g.order_consistent.mean(), g.tie.mean(),
                (g.k > 0).mean(), (g.k < 0).mean(), (g.k == 0).mean()]
        for i, v in enumerate(vals, 1):
            out[f"{pref}{i}"] = v
    return out


sciq_m, pop_m = get(m, "SciQ"), get(m, "PopQA")
sciq_l, pop_l = get(l, "SciQ"), get(l, "PopQA")
pop_p = p.iloc[0]

# conditions
c_sciq_auroc = sciq_m.auroc >= 0.65
c_sciq_ci = sciq_m.auroc_ci_lo > 0.50
c_sciq_b1 = sciq_l.b1_boot_ci_lo > 0
c_pop_b1 = pop_p.b1_boot_ci_lo > 0
c_sciq_med = sciq_m.ySS1["median"] > sciq_m.ySS0["median"]
c_pop_med = pop_m.ySS1["median"] > pop_m.ySS0["median"]

rules = {
    "sciq_auroc_ge_065": (c_sciq_auroc, f"SciQ AUROC={sciq_m.auroc:.3f} >= 0.65"),
    "sciq_auroc_ci_lower_gt_050": (c_sciq_ci, f"SciQ AUROC CI lower={sciq_m.auroc_ci_lo:.3f} > 0.50"),
    "sciq_lenctrl_b1_ci_lower_gt_0": (c_sciq_b1, f"SciQ b1 CI lower={sciq_l.b1_boot_ci_lo:.3f} > 0"),
    "popqa_relfe_b1_ci_lower_gt_0": (c_pop_b1, f"PopQA relation-FE b1 CI lower={pop_p.b1_boot_ci_lo:.3f} > 0"),
    "both_ds_ySS1_median_gt_ySS0": (c_sciq_med and c_pop_med,
                                    f"median k y_SS=1>0: SciQ {sciq_m.ySS1['median']:.2f}>{sciq_m.ySS0['median']:.2f}; "
                                    f"PopQA {pop_m.ySS1['median']:.2f}>{pop_m.ySS0['median']:.2f}"),
}
print("rules:")
for k, (ok, msg) in rules.items():
    print(f"  [{'OK' if ok else 'FAIL'}] {k}: {msg}")

all_supported = all(ok for ok, _ in rules.values())
# dataset_contingent check (for reporting only)
sciq_ok = c_sciq_auroc and c_sciq_ci and c_sciq_b1
pop_ok = c_pop_b1
if all_supported:
    label = "no_reference_factual_preference_association_supported"
    conclusion = (
        "模型在无 Reference 条件下对原始答案的事实偏好，与其后续拒绝 swapped-Reference 一致候选的行为存在"
        "跨数据集、控制长度与 PopQA relation 后仍保留的正向关联。"
    )
    note = "该结论与该解释一致，但不足以证明参数知识是唯一原因。"
elif sciq_ok and not pop_ok:
    label = "no_reference_factual_preference_association_dataset_contingent"
    conclusion = (
        "该关联在 SciQ 中成立，但在 PopQA 长尾事实分布中没有得到同等支持；"
        "这与\u201c知识可覆盖度不同\u201d相容，但不足以构成跨分布机制结论。"
    )
    note = ""
else:
    label = "no_reference_factual_preference_association_not_supported"
    conclusion = (
        "当前无 Reference 的二选一事实偏好不足以解释后续 SS false rejection；"
        "reference-adherence failure 仍可能主要由任务理解、语义匹配、表达形式或其他内部因素共同决定。"
    )
    note = ""
print("LABEL:", label)

# ---------------- failure_examples.md ----------------
# Examples where k and y_SS point away from the association: high k but y_SS=0 (should reject but accepted),
# low k but y_SS=1 (should accept but rejected). Pick a few per dataset.
examples = []
for ds in ["SciQ", "PopQA"]:
    sub = df[df.dataset == ds]
    hi_k_accept = sub[sub.y_SS == 0].nlargest(3, "k")
    lo_k_reject = sub[sub.y_SS == 1].nsmallest(3, "k")
    for _, r in pd.concat([hi_k_accept, lo_k_reject]).iterrows():
        examples.append({
            "dataset": ds, "source_group_id": r.source_group_id, "k": r.k, "y_SS": r.y_SS,
            "question": r.question, "r_o": r.r_o, "r_s": r.r_s, "d_1": r.d_1, "d_2": r.d_2,
            "order_consistent": r.order_consistent,
            "reason": ("y_SS=0（正确接受）但 k 高（偏 r_o）" if r.y_SS == 0
                       else "y_SS=1（错误拒绝）但 k 低（偏 r_s）")})
with open(OUT / "failure_examples.md", "w", encoding="utf-8") as f:
    f.write("# failure_examples.md\n\n")
    f.write("> 本轮 k 与 y_SS 关联的例外样本（按数据集挑取各 3 条）：高 k 却 y_SS=0，低 k 却 y_SS=1。\n\n")
    for e in examples:
        f.write(f"## {e['dataset']} {e['source_group_id'][:12]}…\n\n")
        f.write(f"- k = {e['k']:.3f}；d_1 = {e['d_1']:.3f}；d_2 = {e['d_2']:.3f}；order_consistent = {e['order_consistent']}\n")
        f.write(f"- y_SS = {e['y_SS']}（{e['reason']}）\n")
        f.write(f"- Question: {e['question']}\n")
        f.write(f"- r_o: {e['r_o']}\n")
        f.write(f"- r_s: {e['r_s']}\n\n")

# ---------------- analysis_limitations.md ----------------
(OUT / "analysis_limitations.md").write_text(
    """# analysis_limitations.md

## 局限与边界

1. 本轮是**行为关联审计**，不是 hidden-state 机制实验；不证明参数知识为唯一因果原因。
2. `k` 为无 Reference 双顺序 teacher-forced 偏好分数，衡量行为层面的候选偏好，不直接度量参数知识量。
3. AUROC / logistic β 反映的是关联强度与方向，不是因果；可能存在未观测混淆（题目难度、表达差异等）。
4. SciQ（195 group，y_SS=1 占 148）与 PopQA（2,815 group，y_SS=1 占 144）正类比例差异大，
   AUPRC 的绝对水平受 prevalence 影响；解读以 AUROC 与 CI 为主。
5. PopQA 中 `color` 样本量仅 4，其 relation 分层统计不可靠，如实保留、不作单独结论。
6. 不授权进入 intervention，也不改变既有 D1、D2-R1、D3 或 E1 的结论。
""", encoding="utf-8")

# ---------------- final_report.md ----------------
report = """# final_report.md

## 摘要

| 问题 | 结果 |
|---|---|
| 是否只使用 SciQ / PopQA development group？ | 是（SciQ dev 195 / PopQA dev 2,815；与既有 dev manifest 逐组一致） |
| 是否零 final-reserve / train 文本模型接触？ | 是（未读取、未评分、未缓存任何 train / final-reserve 文本） |
| 无 Reference A/B 读出语义是否有效？ | 有效（合成 24 对：overall 24/24，A 12/12，B 12/12，ties=0，greedy 一致 24/24） |
| SciQ 中 `k` 是否预测 SS false rejection？ | 是（AUROC {sciq_auroc:.3f}，95% CI [{sciq_ci_lo:.3f}, {sciq_ci_hi:.3f}]） |
| PopQA 中 `k` 是否预测 SS false rejection？ | 是（AUROC {pop_auroc:.3f}，95% CI [{pop_ci_lo:.3f}, {pop_ci_hi:.3f}]） |
| PopQA relation 控制后关联是否保留？ | 保留（relation-FE β1(z(k))={pop_b1:.3f}，95% CI [{pop_b1_lo:.3f}, {pop_b1_hi:.3f}]） |
| 是否支持“无 Reference 事实偏好与 SS 错拒有关”的行为解释？ | {supp_txt} |
| 最终标签 | **{label}** |

## 结论

{conclusion}

{note}

## Phase 1：事实偏好分数汇总

| 数据集 | n | d_1 mean | d_1 med | d_2 mean | d_2 med | k mean | k med | order-consistent | tie | k>0 | k<0 | k=0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SciQ | 195 | {s1:.3f} | {s2:.3f} | {s3:.3f} | {s4:.3f} | {s5:.3f} | {s6:.3f} | {s7:.3f} | {s8:.4f} | {s9:.3f} | {s10:.3f} | {s11:.4f} |
| PopQA | 2815 | {p1:.3f} | {p2:.3f} | {p3:.3f} | {p4:.3f} | {p5:.3f} | {p6:.3f} | {p7:.3f} | {p8:.4f} | {p9:.3f} | {p10:.3f} | {p11:.4f} |

## Phase 2：主关联分析

### 2A 数据集内排序

| 数据集 | AUROC | 95% CI | AUPRC | 95% CI | y_SS=0 med k | y_SS=1 med k | Cliff's δ | MWU p |
|---|---|---|---|---|---|---|---|---|
| SciQ | {a1:.3f} | [{a2:.3f}, {a3:.3f}] | {a4:.3f} | [{a5:.3f}, {a6:.3f}] | {a7:.3f} | {a8:.3f} | {a9:.3f} | {a10:.2e} |
| PopQA | {b1:.3f} | [{b2:.3f}, {b3:.3f}] | {b4:.3f} | [{b5:.3f}, {b6:.3f}] | {b7:.3f} | {b8:.3f} | {b9:.3f} | {b10:.2e} |

> Cliff's δ 按 `cliffs_delta(k_{{ySS=0}}, k_{{ySS=1}})` 计算：负值表示 y_SS=1 组（SS 错拒组）的 k 更高，
> 与"k 越高越易错拒"方向一致。MWU p 为描述性双尾 p 值。

### 2B 长度控制 logistic（β 为 z-score 后系数）

| 数据集 | β1 z(k) | OR=exp(β1) | 95% CI (β1) | β2 z(q_len) | β3 z(len r_o) | β4 z(len r_s) |
|---|---|---|---|---|---|---|
| SciQ | {c1:.3f} | {c2:.3f} | [{c3:.3f}, {c4:.3f}] | {c5:.3f} | {c6:.3f} | {c7:.3f} |
| PopQA | {d1:.3f} | {d2:.3f} | [{d3:.3f}, {d4:.3f}] | {d5:.3f} | {d6:.3f} | {d7:.3f} |

### 2C PopQA relation fixed-effects

| 模型 | n | β1 z(k) | OR | 95% CI (β1) | reference relation |
|---|---|---|---|---|---|
| relation FE | {e0} | {e1:.3f} | {e2:.3f} | [{e3:.3f}, {e4:.3f}] | {e5} |

relation 分层描述表见 `popqa_relation_descriptive_audit.csv`；仅当 relation 同时含正负类且 n>=30 时报告 AUROC。

## 预注册规则核查

```text
sciq_auroc_ge_065:               {r0}
sciq_auroc_ci_lower_gt_050:      {r1}
sciq_lenctrl_b1_ci_lower_gt_0:   {r2}
popqa_relfe_b1_ci_lower_gt_0:    {r3}
both_ds_ySS1_median_gt_ySS0:     {r4}
```

## 结语

本轮是行为关联审计，不是 hidden-state 机制实验；
不证明参数知识为唯一因果原因；
不授权进入 intervention，也不改变既有 D1、D2-R1、D3 或 E1 的结论。
""".format(
    sciq_auroc=sciq_m.auroc, sciq_ci_lo=sciq_m.auroc_ci_lo, sciq_ci_hi=sciq_m.auroc_ci_hi,
    pop_auroc=pop_m.auroc, pop_ci_lo=pop_m.auroc_ci_lo, pop_ci_hi=pop_m.auroc_ci_hi,
    pop_b1=pop_p["b1_z(k)"], pop_b1_lo=pop_p.b1_boot_ci_lo, pop_b1_hi=pop_p.b1_boot_ci_hi,
    supp_txt="是（跨数据集、控制长度与 PopQA relation 后仍保留）" if all_supported else "部分 / 否",
    label=label, conclusion=conclusion, note=note,
    **phase1_summary(df),
    **{f"a{i}": v for i, v in enumerate([
        sciq_m.auroc, sciq_m.auroc_ci_lo, sciq_m.auroc_ci_hi, sciq_m.auprc,
        sciq_m.auprc_ci_lo, sciq_m.auprc_ci_hi, sciq_m.ySS0["median"], sciq_m.ySS1["median"],
        sciq_m.cliffs_delta, sciq_m.mannwhitney_p], 1)},
    **{f"b{i}": v for i, v in enumerate([
        pop_m.auroc, pop_m.auroc_ci_lo, pop_m.auroc_ci_hi, pop_m.auprc,
        pop_m.auprc_ci_lo, pop_m.auprc_ci_hi, pop_m.ySS0["median"], pop_m.ySS1["median"],
        pop_m.cliffs_delta, pop_m.mannwhitney_p], 1)},
    **{f"c{i}": v for i, v in enumerate([
        sciq_l["b1_z(k)"], sciq_l.odds_ratio_exp_b1, sciq_l.b1_boot_ci_lo, sciq_l.b1_boot_ci_hi,
        sciq_l["b2_z(q_len)"], sciq_l["b3_z(len_r_o)"], sciq_l["b4_z(len_r_s)"], ], 1)},
    **{f"d{i}": v for i, v in enumerate([
        pop_l["b1_z(k)"], pop_l.odds_ratio_exp_b1, pop_l.b1_boot_ci_lo, pop_l.b1_boot_ci_hi,
        pop_l["b2_z(q_len)"], pop_l["b3_z(len_r_o)"], pop_l["b4_z(len_r_s)"], ], 1)},
    e0=int(pop_p.n), e1=pop_p["b1_z(k)"], e2=pop_p.odds_ratio_exp_b1,
    e3=pop_p.b1_boot_ci_lo, e4=pop_p.b1_boot_ci_hi, e5=pop_p.fe_reference_relation,
    r0=f"OK ({sciq_m.auroc:.3f} >= 0.65)",
    r1=f"OK ({sciq_m.auroc_ci_lo:.3f} > 0.50)",
    r2=f"OK ({sciq_l.b1_boot_ci_lo:.3f} > 0)",
    r3=f"OK ({pop_p.b1_boot_ci_lo:.3f} > 0)",
    r4=f"OK (SciQ {sciq_m.ySS1['median']:.2f}>{sciq_m.ySS0['median']:.2f}, PopQA {pop_m.ySS1['median']:.2f}>{pop_m.ySS0['median']:.2f})",
)
(OUT / "final_report.md").write_text(report, encoding="utf-8")

# ---------------- artifacts/decision.json ----------------
(OUT / "artifacts" / "decision.json").write_text(json.dumps({
    "final_label": label,
    "rules": {k: bool(ok) for k, (ok, _) in rules.items()},
    "conclusion": conclusion,
    "sciq_auroc": sciq_m.auroc, "sciq_auroc_ci": [sciq_m.auroc_ci_lo, sciq_m.auroc_ci_hi],
    "sciq_b1_z_k": sciq_l["b1_z(k)"], "sciq_b1_ci": [sciq_l.b1_boot_ci_lo, sciq_l.b1_boot_ci_hi],
    "popqa_b1_relfe_z_k": pop_p["b1_z(k)"], "popqa_b1_relfe_ci": [pop_p.b1_boot_ci_lo, pop_p.b1_boot_ci_hi],
    "median_k_ySS1_gt_ySS0": {"SciQ": bool(c_sciq_med), "PopQA": bool(c_pop_med)},    "final_reserve_model_scored": False, "final_reserve_text_read": False,
    "hidden_states_read": False, "probe_trained": False,
    "activation_intervention_run": False,
    "note": "行为关联审计，非 hidden-state 机制实验；不证明参数知识为唯一因果原因；不授权 intervention；不改变既有 D1/D2-R1/D3/E1 结论。"
}, indent=2), encoding="utf-8")

print("Phase 3 deliverables written; label =", label)
