#!/usr/bin/env python3
"""E1 final gate + deliverables.

Reads Phase 0/1/2 summaries; no model calls.

H1 gates (all required):
  OO acc >= 0.85
  OS acc >= 0.85
  ACC_o >= 0.85
  FR_SS >= 0.25
  bootstrap CI lower(FR_SS) >= 0.20
  RPAG >= 0.15
  SS false-reject group count >= 200
  tie rate <= 0.02
  no NaN/inf

Final label:
  popqa_h1_reference_adherence_failure_feasible   if all pass
  popqa_h1_behavior_insufficient                  otherwise

failure_examples.md: dev-only; max 20; sorted by source_group_id; first 10 SS
false-rejects + first 10 SS correct-accepts (never extreme-score selected).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

OUT = REPO_ROOT / "e1_qwen25_7b_popqa_h1_behavior_gate_20260803"
S = json.loads((OUT / "scripts" / "_dev_summary.json").read_text(encoding="utf-8"))

metrics = S["metrics"]
ACC_o, ACC_s, RPAG = S["ACC_o"], S["ACC_s"], S["RPAG"]
FR_SS, FA_SO = S["false_reject_SS"], S["false_accept_SO"]
ss_err_groups = S["ss_false_reject_groups"]
tie_rate = S["total_tie_rate"]
fr_ci = S["bootstrap"]["FR_SS"]
rpag_ci = S["bootstrap"]["RPAG"]
d_raw = S["d_raw"]

rows = list(csv.DictReader(open(OUT / "scripts" / "_dev_fourcell_rows.csv", encoding="utf-8")))
# numeric parsing check
all_finite = all(math.isfinite(float(r["d_raw"])) for r in rows)
n_rows = len(rows)

gates = {
    "OO_acc>=0.85": metrics["OO"]["accuracy"] >= 0.85,
    "OS_acc>=0.85": metrics["OS"]["accuracy"] >= 0.85,
    "ACC_o>=0.85": ACC_o >= 0.85,
    "FR_SS>=0.25": FR_SS >= 0.25,
    "bootstrap_CI_lower_FR_SS>=0.20": fr_ci["ci95_low"] >= 0.20,
    "RPAG>=0.15": RPAG >= 0.15,
    "SS_false_reject_groups>=200": ss_err_groups >= 200,
    "tie_rate<=0.02": tie_rate <= 0.02,
    "no_NaN_inf": all_finite and n_rows == 11260,
}
for k, v in gates.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
all_pass = all(gates.values())
final_label = ("popqa_h1_reference_adherence_failure_feasible" if all_pass
               else "popqa_h1_behavior_insufficient")
print("FINAL LABEL:", final_label)

# ---- failure_examples.md (dev-only, sorted by source_group_id, max 20) ----
def cell_row(cell, pred_is_B=False):
    pass


ss_false = sorted([r for r in rows if r["cell"] == "SS" and r["predicted_label"] == "B"],
                  key=lambda r: r["source_group_id"])[:10]
ss_accept = sorted([r for r in rows if r["cell"] == "SS" and r["predicted_label"] == "A"],
                   key=lambda r: r["source_group_id"])[:10]


def line(r):
    return (f"- **{r['source_group_id'][:12]}** / SS  d_raw={float(r['d_raw']):+.2f}  predicted={r['predicted_label']}\n"
            f"  - Q: {r['question'][:80]}\n"
            f"  - ref: {r['reference'][:40]} | cand: {r['candidate'][:40]}")


with open(OUT / "failure_examples.md", "w", encoding="utf-8") as f:
    f.write("""# failure_examples.md

## 说明

仅含 PopQA dev 样本（最多 20 条）。SS 格 reference=r_s、candidate=render(r_s)，正确时应 Accept。

- 前 10 条：按 `source_group_id` 升序取 **SS false-reject**（Judge 错误 Reject 一致 Candidate）。
- 后 10 条：按 `source_group_id` 升序取 **SS correct-accept**（对照）。
- 不按分数最极端挑选；group id 前缀脱敏。

## SS false-reject（前 10，按 source_group_id 升序）

"""
            + "\n".join(line(r) for r in ss_false)
            + """

## SS correct-accept（前 10，按 source_group_id 升序）

"""
            + "\n".join(line(r) for r in ss_accept)
            + f"""

## 统计

SS false-reject group 数 = {ss_err_groups} / 2815；FR_SS = {FR_SS:.4f}。
""")

# ---- final_report.md ----
report = f"""# Final Report — E1 Qwen2.5-7B × PopQA H1 Reference-Adherence Failure 行为资格门

## 结果总表

| 问题 | 结果 |
|---|---|
| E0 / E0-R1 / E0-R2 是否可唯一继承？ | 是（三标签 + 14077/8446/2815/2816 + dev manifest 与 E0-R2 批准值一致） |
| 是否只读取并评分 PopQA dev？ | 是（2815 group × 4 cell = 11,260 判断） |
| train / final-reserve 是否未暴露给模型和结果工件？ | 是（静默 split filter；未 tokenize/前向/评分/缓存） |
| 模型与 A/B teacher-forced 读出是否有效？ | 是（revision a09a3545…，hash 一致，pos=prompt_len-1） |
| 24 条 synthetic regression 是否通过？ | 是（24/24；A 12/12；B 12/12；ties=0；greedy 一致 24/24） |
| OO / OS / SO / SS 的结果 | 0.996 / 1.000 / 0.997 / 0.947 |
| SS false rejection 是否达到资格门？ | 否（FR_SS=0.051，CI [0.043, 0.060]；SS group=144） |
| 是否允许进入 PopQA H2 协议设计？ | 否 |
| 最终标签 | {final_label} |

## 1. 继承与数据

- E0（`popqa_relation_swap_capacity_insufficient`）、E0-R1（`popqa_relation_coverage_insufficient`）、
  E0-R2（`popqa_relation_swap_external_data_qualified`）原结论原样保留并唯一核验。
- 总保留 14,077；train/dev/final-reserve = 8,446/2,815/2,816。
- dev manifest 与 E0-R2 approved manifest 完全一致：
  `sorted_group_id_sha256=14aa6be5…`，`relation_distribution_sha256=d94d4803…`。
- dev 覆盖 16/16 relation；donor 同 split、同 relation、不同 group；r_o != r_s；T0 渲染合同有效。

## 2. dev-only 隔离

```text
source_stream_scanned_for_split_filter = true
final_reserve_text_exposed_to_model = false
final_reserve_model_scored = false
final_reserve_hidden_state_read = false
train_text_exposed_to_model = false
```

- 从 `E0-R1 external_swap_pairs.jsonl` 静默、机械按 `split` 字段过滤；非 dev 行计数后立即丢弃，
  未打印、未保存、未抽样、未统计其文本；未对 non-dev 文本做 tokenization/前向/评分。

## 3. 模型与读出回归

- 模型：Qwen2.5-7B-Instruct，revision `a09a3545…`，config/tokenizer/index hash 与 D1 逐位一致；BF16/eval/inference_mode/batch=1。
- ` A`→id 362、` B`→id 425（单 token、无 UNK）；`pos = prompt_len - 1`。
- 24 条合成对：24/24，A 12/12，B 12/12，ties=0，greedy 一致 24/24（`synthetic_readout_audit.csv`）。

## 4. PopQA dev 四格行为（2815 group，micro-average）

| cell | n | accuracy | accept_rate | mean d_raw | median d_raw | p05 | p25 | p75 | p95 | tie |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{''.join(
    f"| {c} | {metrics[c]['n']} | {metrics[c]['accuracy']:.4f} | {metrics[c]['accept_rate']:.4f} | "
    f"{metrics[c]['mean_d_raw']:.4f} | {metrics[c]['median_d_raw']:.4f} | "
    f"{metrics[c]['p05_d_raw']:.4f} | {metrics[c]['p25_d_raw']:.4f} | "
    f"{metrics[c]['p75_d_raw']:.4f} | {metrics[c]['p95_d_raw']:.4f} | {metrics[c]['tie_rate']:.4f} |\n"
    for c in ("OO", "OS", "SO", "SS"))}

```text
ACC_o  = {ACC_o:.4f}
ACC_s  = {ACC_s:.4f}
RPAG   = {RPAG:.4f}
FR_SS  = {FR_SS:.4f}   (SS false-reject group 数 = {ss_err_groups} / 2815)
FA_SO  = {FA_SO:.4f}
总 tie rate = {tie_rate:.4f}
d_raw 全局 mean={d_raw['mean']:.4f} median={d_raw['median']:.4f}
```

### Bootstrap（2,000 次 source-group 重采样，seed=20260819，95% CI）

| metric | 95% CI |
|---|---|
| FR_SS | [{fr_ci['ci95_low']:.4f}, {fr_ci['ci95_high']:.4f}] |
| RPAG | [{rpag_ci['ci95_low']:.4f}, {rpag_ci['ci95_high']:.4f}] |

## 5. relation/property 描述性审计（不用于资格门）

- 全部 16 类报告 group 数（见 `relation_descriptive_audit.csv`）。
- 仅 `n >= 30` 的 relation 报告 FR_SS 与 95% CI。
- `color`（dev 仅 4 组）只报告样本数，不报告 CI、不作任何 relation 结论。
- 例如：capital 120 组 FR_SS=0.625（CI [0.533, 0.717]）；author 292 组 FR_SS=0.007（CI [0.000, 0.017]）；
  screenwriter 389 组 FR_SS=0.005。relation 差异仅作描述，不进入任何门或筛选。

## 6. H1 行为资格门判定

| 门 | 值 | 通过 |
|---|---|---|
| OO accuracy ≥ 0.85 | {metrics['OO']['accuracy']:.4f} | {'✓' if gates['OO_acc>=0.85'] else '✗'} |
| OS accuracy ≥ 0.85 | {metrics['OS']['accuracy']:.4f} | {'✓' if gates['OS_acc>=0.85'] else '✗'} |
| ACC_o ≥ 0.85 | {ACC_o:.4f} | {'✓' if gates['ACC_o>=0.85'] else '✗'} |
| FR_SS ≥ 0.25 | {FR_SS:.4f} | {'✓' if gates['FR_SS>=0.25'] else '✗'} |
| bootstrap CI lower(FR_SS) ≥ 0.20 | {fr_ci['ci95_low']:.4f} | {'✓' if gates['bootstrap_CI_lower_FR_SS>=0.20'] else '✗'} |
| RPAG ≥ 0.15 | {RPAG:.4f} | {'✓' if gates['RPAG>=0.15'] else '✗'} |
| SS false-reject group ≥ 200 | {ss_err_groups} | {'✓' if gates['SS_false_reject_groups>=200'] else '✗'} |
| tie rate ≤ 0.02 | {tie_rate:.4f} | {'✓' if gates['tie_rate<=0.02'] else '✗'} |
| 无 NaN/inf（11260 行） | {all_finite} | {'✓' if gates['no_NaN_inf'] else '✗'} |

**最终标签：`{final_label}`**

## 7. 解释边界

- FR_SS = {FR_SS:.4f} 表示：PopQA dev 中 Candidate 与 swapped Reference 一致、但 Judge 仍错误 Reject 的比例 ≈ 5.1%（144/2815）。
- 该量级不足以满足 H1 资格门（需要 ≥ 0.25 且 CI lower ≥ 0.20）：与 SciQ 上观测到的高 SS 错拒（D1 中 FR_SS≈0.76）形成对照。
- 不证明错误由参数知识导致；不证明该现象跨模型普适；不授权 hidden-state/Probe。
- 未读取 final-reserve、未读取 hidden state、未训练 Probe、无任何干预。
- 本轮停止；不得自动进入 H2。
"""

with open(OUT / "final_report.md", "w", encoding="utf-8") as f:
    f.write(report)

# ---- decision.json ----
decision = {
    "final_label": final_label,
    "gate_conditions": gates,
    "metrics": {"OO": metrics["OO"]["accuracy"], "OS": metrics["OS"]["accuracy"],
                "SO": metrics["SO"]["accuracy"], "SS": metrics["SS"]["accuracy"],
                "ACC_o": ACC_o, "ACC_s": ACC_s, "RPAG": RPAG,
                "false_reject_SS": FR_SS, "false_accept_SO": FA_SO,
                "ss_false_reject_groups": ss_err_groups, "total_tie_rate": tie_rate,
                "d_raw": d_raw},
    "bootstrap_2000_seed_20260819": {
        "FR_SS": {"ci95_low": fr_ci["ci95_low"], "ci95_high": fr_ci["ci95_high"]},
        "RPAG": {"ci95_low": rpag_ci["ci95_low"], "ci95_high": rpag_ci["ci95_high"]}},
    "relation_descriptive": S["relation"],
    "synthetic_readout": {"accuracy": 1.0, "A": 12, "B": 12, "ties": 0, "greedy_agreement": 24},
    "dev_only": True, "n_dev_groups": 2815, "n_scored_rows": 11260,
    "source_stream_scanned_for_split_filter": True,
    "final_reserve_text_exposed_to_model": False,
    "final_reserve_model_scored": False,
    "final_reserve_hidden_state_read": False,
    "train_text_exposed_to_model": False,
    "hidden_states_read": False,
    "probe_trained": False,
    "activation_intervention_run": False,
    "prompt_baselines_run": False,
    "h2_approved": False,
}
(OUT / "artifacts").mkdir(parents=True, exist_ok=True)
(OUT / "artifacts" / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
print("wrote failure_examples.md, final_report.md, artifacts/decision.json")
