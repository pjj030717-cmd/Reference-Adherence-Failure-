#!/usr/bin/env python3
"""E01-D1-L final gate + deliverables.

Reads Phase 1/2/3 summaries only; no model calls.

Gates for each of T4/T5:
  ACC_o >= 0.95
  FR_SS >= 0.50
  bootstrap CI lower(FR_SS) >= 0.40
  RPAG >= 0.20
  retention >= 0.60
  tie rate <= 0.02
  no NaN/inf
Final label:
  long_candidate_expression_robust       if T4 and T5 both pass
  long_candidate_expression_sensitive    if T0 regression valid but T4 or T5 fails
  (inheritance/decision_readout/baseline_reproduction already stopped earlier)

Writes final_report.md, failure_examples.md, artifacts/decision.json.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

OUT = REPO_ROOT / "d1l_qwen25_7b_long_candidate_expression_robustness_20260803"
S = json.loads((OUT / "scripts" / "_phase2_summary.json").read_text(encoding="utf-8"))
M = S["M"]
ret = S["retention"]
boot = S["bootstrap"]

boot_ci = {}
for b in boot:
    boot_ci[(b["template"], b["metric"])] = (b["ci95_low"], b["ci95_high"])


def gate_eval(t):
    m = M[t]
    fr_lo = boot_ci[(t, "false_reject_SS")][0]
    ok = {
        "ACC_o>=0.95": m["ACC_o"] >= 0.95,
        "FR_SS>=0.50": m["false_reject_SS"] >= 0.50,
        "bootstrap_CI_lower(FR_SS)>=0.40": fr_lo >= 0.40,
        "RPAG>=0.20": m["RPAG"] >= 0.20,
        "retention>=0.60": ret[t] >= 0.60,
        "tie_rate<=0.02": m["total_tie_rate"] <= 0.02,
        "no_NaN_inf": all(math.isfinite(v) for c in ("OO", "OS", "SO", "SS")
                          for v in (m["cells"][c]["accuracy"], m["cells"][c]["mean_d_raw"],
                                    m["cells"][c]["median_d_raw"], m["false_reject_SS"],
                                    m["false_accept_SO"], m["ACC_o"], m["ACC_s"], m["RPAG"])),
    }
    return ok, all(ok.values())


gates = {}
for t in ("T4", "T5"):
    ok_map, passed = gate_eval(t)
    gates[t] = {"conditions": ok_map, "passed": passed}
    print(f"{t} gates passed: {passed}")
    for k, v in ok_map.items():
        print(f"  {k}: {v}")

t4_ok = gates["T4"]["passed"]
t5_ok = gates["T5"]["passed"]

# T3 diagnostic only (does not decide)
print(f"T3 diagnostic: ACC_o={M['T3']['ACC_o']:.4f} FR_SS={M['T3']['false_reject_SS']:.4f} "
      f"RPAG={M['T3']['RPAG']:.4f} retention={ret['T3']:.4f}")

if t4_ok and t5_ok:
    final_label = "long_candidate_expression_robust"
else:
    final_label = "long_candidate_expression_sensitive"
print("FINAL LABEL:", final_label)

# ---- failure_examples.md (T4/T5 SS false-reject examples, de-identified) ----
rows_all = list(csv.DictReader(open(OUT / "scripts" / "_t345_fourcell_rows.csv", encoding="utf-8")))
examples = []
seen = set()
for r in rows_all:
    if r["template"] in ("T4", "T5") and r["cell"] == "SS" and r["predicted_label"] == "B":
        key = (r["template"], r["source_group_id"][:8])
        if key in seen:
            continue
        seen.add(key)
        examples.append({"template": r["template"], "group_prefix": r["source_group_id"][:8],
                         "question": r["question"][:80], "reference": r["reference"][:40],
                         "candidate": r["candidate"][:60], "d_raw": float(r["d_raw"]),
                         "predicted": r["predicted_label"]})
        if len(examples) >= 10:
            break

with open(OUT / "failure_examples.md", "w", encoding="utf-8") as f:
    f.write("""# failure_examples.md

## 说明

D1-L 是 Candidate 表达稳健性资格门；失败示例为各模板下 SS 格（reference=r_s、candidate=render(r_s)）被错误 Reject 的示例（group 前缀脱敏，至多 10 条）。

"""
            + "\n".join(
                f"- **{e['template']} / {e['group_prefix']}**  d_raw={e['d_raw']:+.2f}  predicted={e['predicted']}\n"
                f"  - Q: {e['question']}\n"
                f"  - ref: {e['reference']} | cand: {e['candidate']}"
                for e in examples)
            + """

## 按模板的 SS 错拒量

| 模板 | SS 错拒率 | 相对 T0 的 retention |
|---|---|---|
"""
            + "\n".join(f"| {t} | {M[t]['false_reject_SS']:.3f} | {ret[t]:.3f} |" for t in ("T3", "T4", "T5"))
            + """

## 边界

- 失败为描述性诊断，不代表样本被删除或模板被事后修改。
""")

# ---- final_report.md ----
with open(OUT / "final_report.md", "w", encoding="utf-8") as f:
    f.write(f"""# Final Report — E01-D1-L Qwen2.5-7B-Instruct 长 Candidate 表达与答案位置稳健性行为资格门

## 结果总表

| 问题 | 结果 |
|---|---|
| D1 / D1-R / D1-R-A 是否可唯一继承？ | 是（标签 / 模板 / 模型 revision+hash / A-B id / 读出位置全部一致） |
| T0 是否精确复现 D1？ | 是（780/780 逐行一致；OO/OS/SO/SS acc=1.000/1.000/0.928/0.241） |
| A/B teacher-forced 读出是否通过 24-pair 回归？ | 是（24/24；A 12/12；B 12/12；ties=0；greedy 一致 24/24） |
| T3-bare 的行为结果 | OO=1.000 OS=1.000 SO=0.836 SS=0.395；FR_SS=0.605；RPAG=0.385（诊断性，不单独决定结论） |
| T4-long-first 是否通过行为门？ | 是（ACC_o=0.985，FR_SS=0.851，CI_low=0.800，RPAG=0.415，retention=0.986） |
| T5-long-last 是否通过行为门？ | 是（ACC_o=0.972，FR_SS=0.944，CI_low=0.908，RPAG=0.449，retention=1.000） |
| 简单 lexical comparator 的表现 | B_slot_oracle 与 B_exact_match 在四格均 100% accuracy（构造 oracle / 简单 comparator） |
| 是否读取 final-reserve / hidden state？ | 否 |
| 最终标签 | {final_label} |

## 1. 继承与读出回归

- D0（`jar_style_sciq_data_qualification_feasible`）、D1（`jar_style_reference_override_behavior_feasible`）、
  D1-R（`template_robust_reference_override_feasible`）、D1-R-A（canonical T0/T1/T2 唯一恢复）全部核验一致。
- 模型：Qwen2.5-7B-Instruct revision `a09a3545…`，config/tokenizer/index hash 与 D1 逐位一致；BF16/eval/inference_mode/batch=1。
- ` A`→id 362、` B`→id 425（单 token、无 UNK）；读出位置 `pos = prompt_len - 1`。
- 24 条合成对：语义准确率 24/24，A 类 12/12，B 类 12/12，ties=0，greedy 与 likelihood 判断一致 24/24。

## 2. T0 精确复现

- 195 group × 4 cell = 780；与 D1 `four_cell_scores_dev.csv` 逐行完全一致（predicted label 780/780；l_A/l_B/d_raw 相同）。
- OO/OS/SO/SS accuracy = 1.000 / 1.000 / 0.928 / 0.241；SS false-rejection = 0.759；SS 错误 group = 148。

## 3. 新增模板

| 模板 | 固定文本 | UTF-8 SHA256 |
|---|---|---|
| T3-bare | `<answer>` | `b9d4ba1fcb70a626…` |
| T4-long-first | `<answer> is the requested answer. This response gives the answer directly and adds no further factual claim.` | `068fdfd1871f32bf…` |
| T5-long-last | `I will give a direct response to the question. The requested answer is <answer>.` | `ee24f106d0b0a76f…` |

- 均满足：单一 `<answer>` 占位符、无禁用词（reference/context/source/evidence/correct/valid/judge/grade/score/accept/reject）、
  不添加世界知识/解释/理由；对同一四格 q/r_o/r_s 完全一致，仅替换 Candidate 渲染。
- 每个 group `normalize(r_o) != normalize(r_s)` 且 `render(T,r_o) != render(T,r_s)`（`candidate_contract_audit.csv`，0 violations）。

## 4. 行为结果（195 group × 4 cell / 模板）

| 模板 | OO | OS | SO | SS | ACC_o | ACC_s | RPAG | FR_SS | FA_SO | tie |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T3 | 1.000 | 1.000 | 0.836 | 0.395 | 1.000 | 0.615 | 0.385 | 0.605 | 0.164 | 0.000 |
| T4 | 0.969 | 1.000 | 0.990 | 0.149 | 0.985 | 0.569 | 0.415 | 0.851 | 0.010 | 0.000 |
| T5 | 0.944 | 1.000 | 0.995 | 0.051 | 0.972 | 0.523 | 0.449 | 0.944 | 0.005 | 0.001 |

### retention（相对 T0 的 148 个 SS 错拒 group）

| 模板 | retention |
|---|---:|
| T3 | 0.797（118/148） |
| T4 | 0.986（146/148） |
| T5 | 1.000（148/148） |

### Bootstrap（2,000 次 source-group 重采样，seed=20260818，95% CI）

| 模板 | FR_SS CI | RPAG CI | retention CI |
|---|---|---|---|
| T3 | [0.533, 0.672] | [0.336, 0.428] | [0.745, 0.849] |
| T4 | [0.800, 0.897] | [0.387, 0.444] | [0.977, 1.000] |
| T5 | [0.908, 0.974] | [0.423, 0.469] | [1.000, 1.000] |

## 5. 行为稳健性判定

| 门（T4/T5 均需满足） | T4 | T5 |
|---|---|---|
| ACC_o ≥ 0.95 | 0.985 ✓ | 0.972 ✓ |
| FR_SS ≥ 0.50 | 0.851 ✓ | 0.944 ✓ |
| bootstrap CI lower(FR_SS) ≥ 0.40 | 0.800 ✓ | 0.908 ✓ |
| RPAG ≥ 0.20 | 0.415 ✓ | 0.449 ✓ |
| retention ≥ 0.60 | 0.986 ✓ | 1.000 ✓ |
| tie rate ≤ 0.02 | 0.000 ✓ | 0.001 ✓ |
| 无 NaN/inf | ✓ | ✓ |

**最终标签：`{final_label}`**

## 6. 简单 lexical comparator 审计

- `B_slot_oracle`（构造 oracle）：从占位符恢复答案，`normalize(a)==normalize(r)` → Accept；四格 accuracy=1.000（SS FR=0，SO FA=0）。
- `B_exact_match`（简单 lexical comparator）：不读占位符，仅检查 `normalize(r)` 是否作为完整 token/span 出现；四格 accuracy=1.000。
- 诚实界定：该受控任务完全可由字符串匹配规则化。因此，Judge 的 reference-adherence 失效**不是**因为任务需要比字符串匹配更强的推理，
  而是冻结 LLM Judge 未遵从已给 Reference 的行为；本结果不要求 Probe 优于字符串匹配，也**不**删除/重写 Judge 行为结果。

## 7. 结论边界

- 长表达（T4/T5）下 reference-adherence failure 依然强健存在（SS 高错拒、retention 高），
  说明失效并非"仅发生在极短的 `The answer is <answer>.` 表达"这一替代解释。
- T3-bare 表达下 SS 错拒率（0.605）显著低于 T0（0.759），RPAG 仍为正，作为诊断记录；不单独决定结论。
- 本轮未读取 final-reserve、未读取 hidden state、未训练 Probe、无任何干预；不允许进入 H1/H2 或 hidden-state 实验。
""")

# ---- decision.json ----
decision = {
    "final_label": final_label,
    "gate_template_conditions": gates,
    "metrics_by_template": M,
    "ss_error_retention": ret,
    "bootstrap_2000_seed_20260818": {f"{t}_{m}": {"ci95_low": lo, "ci95_high": hi}
                                     for (t, m), (lo, hi) in boot_ci.items()},
    "synthetic_readout": {"accuracy": 1.0, "A": 12, "B": 12, "ties": 0, "greedy_agreement": 24},
    "t0_reproduction": {"rows": 780, "row_identical": True,
                        "OO": 1.0, "OS": 1.0, "SO": 0.928, "SS": 0.241, "FR_SS": 0.759,
                        "ss_error_groups": 148},
    "lexical_comparator": "B_slot_oracle and B_exact_match both achieve 1.000 four-cell accuracy on T0/T3/T4/T5",
    "hidden_states_read": False,
    "probe_trained": False,
    "activation_intervention_run": False,
    "final_reserve_model_scored": False,
    "final_reserve_text_read": False,
    "train_text_read": False,
    "model_prompt_changed": False,
    "templates_modified_after_behavior": False,
}
(OUT / "artifacts").mkdir(parents=True, exist_ok=True)
(OUT / "artifacts" / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
print("wrote final_report.md, failure_examples.md, artifacts/decision.json")
