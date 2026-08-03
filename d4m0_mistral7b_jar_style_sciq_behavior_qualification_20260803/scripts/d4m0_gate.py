#!/usr/bin/env python3
"""D4-M0 final gate: failure_examples.md, final_report.md, artifacts/decision.json.

Phase 3 did NOT pass the D1-R-inherited template robustness gate (FR_SS >= 0.50
for both T1 and T2), so the final label is mistral_template_robustness_insufficient.
No T1/T2 -> feasible label. Phase 4 (true-prefix monitor) is not authorized.

Reads pre-computed summaries only; no model calls, no D0 train/final-reserve.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "d4m0_mistral7b_jar_style_sciq_behavior_qualification_20260803"
P2 = json.loads((R / "scripts" / "_phase2_summary.json").read_text(encoding="utf-8"))
P3 = json.loads((R / "scripts" / "_phase3_summary.json").read_text(encoding="utf-8"))
M = {**{t: P2[t] for t in ["acc_by_cell", "ACC_o", "ACC_s", "RPAG", "false_reject_SS",
                            "false_accept_SO", "ss_error_groups", "so_error_groups",
                            "tie_rate", "nan_inf"]}, **P3["template_summaries"]}

acc_t0 = P2["acc_by_cell"]
acc_t1 = P3["template_summaries"]["T1"]["acc_by_cell"]
acc_t2 = P3["template_summaries"]["T2"]["acc_by_cell"]

gate_ok_T1 = all(v for v in P3["gate_conditions"]["T1"].values())
gate_ok_T2 = all(v for v in P3["gate_conditions"]["T2"].values())
final_label = "mistral_template_robustness_insufficient"
print("gate T1:", gate_ok_T1, "gate T2:", gate_ok_T2)
print("FINAL LABEL:", final_label)

# ---------------------------------------------------------------------------
# failure_examples.md (T0-anchored SS errors, retention per template)
# ---------------------------------------------------------------------------
ret_rows = list(csv.DictReader(open(R / "template_error_retention_audit.csv", encoding="utf-8")))
ex_lines = []
for r in ret_rows[:10]:
    ex_lines.append(
        f"- **{r['source_group_id'][:8]}**\n"
        f"  - Q: {r['question'][:80]}\n"
        f"  - T0_SS d_raw={r['T0_SS_d_raw']} correct={r['T0_SS_correct']} | "
        f"T1 correct={r['T1_SS_correct']} d_raw={r['T1_SS_d_raw']} | "
        f"T2 correct={r['T2_SS_correct']} d_raw={r['T2_SS_d_raw']}")
(R / "failure_examples.md").write_text(
    "# failure_examples.md\n\n"
    "T0 的 SS 错拒 group 在 T1/T2 下的保持情况示例（至多 10 条；group 前缀脱敏）。\n\n"
    + "\n".join(ex_lines)
    + f"\n\nSS_error_retention：T1={P3['retention']['T1']:.3f}（{int(P3['retention']['T1']*P3['t0_ss_error_groups'])}/{P3['t0_ss_error_groups']}），"
      f"T2={P3['retention']['T2']:.3f}（{int(P3['retention']['T2']*P3['t0_ss_error_groups'])}/{P3['t0_ss_error_groups']}）。\n",
    encoding="utf-8")

# ---------------------------------------------------------------------------
# final_report.md
# ---------------------------------------------------------------------------
(R / "final_report.md").write_text(
    f"""# Final Report — D4-M0 Mistral-7B-Instruct-v0.3 JAR-style SciQ 行为资格门

| 问题 | 结果 |
|---|---|
| D0 数据与 dev split 是否唯一继承？ | 是（D0 标签 `jar_style_sciq_data_qualification_feasible`；dev=195 组；swap/四格/T0 渲染语义唯一恢复；无 train/final-reserve 文本进入本目录） |
| Mistral 是否可本地 BF16 加载？ | 是（本地路径，BF16/eval/inference_mode/batch=1，revision 本地不可得；见 `model_access_audit.md`） |
| A/B continuation 是否公平且语义正确？ | 是（` A`→id 1098，` B`→id 1133；单 token、不同 id、无 UNK、等长；24/24 synthetic 回归通过） |
| T0 是否通过 reference-adherence 行为门？ | 是（ACC_o=0.995≥0.85，RPAG=0.341≥0.15，SS 错拒 76 组≥50 且率 0.390≥0.25，ties=0，无 NaN） |
| T1/T2 是否通过模板稳健性门？ | 否（D1-R 继承门槛 FR_SS≥0.50 未满足：T1=0.482，T2=0.456；T2 的 CI lower=0.385<0.40） |
| 是否允许进入 Mistral true-prefix Probe？ | 否 |
| final-reserve 是否完全未触碰？ | 是 |
| 最终标签 | `mistral_template_robustness_insufficient` |

## 1. 继承与模型

- D0：`jar_style_sciq_data_qualification_feasible`；dev split 195 组（seed 20260802，SHA256 `8be6f6f3…`）。
- D1：`jar_style_reference_override_behavior_feasible`；D1-R：`template_robust_reference_override_feasible`。
- 模板：T0/T1/T2 从 D1-R `candidate_template_robustness_spec.json` 逐字继承（SHA256 复核一致）。
- 模型：Mistral-7B-Instruct-v0.3，本地 `/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3`；BF16、eval、inference_mode、batch_size=1；无下载/无替换。
- 唯一序列化变化：Qwen native chat template → Mistral 官方 `apply_chat_template`（见 `prompt_semantic_inheritance_audit.md`）。

## 2. Tokenizer 与读出语义回归（Phase 1）

- continuation：` A`→id 1098，` B`→id 1133；两者均单 token、长度相同、id 不同、无 UNK（见 `tokenization_audit.md`）。
- teacher-forced 位置固定为 `logits[:, prompt_len-1, :]`；无 prior correction / logit bias / 后处理（见 `teacher_forcing_implementation_audit.md`）。
- 24 条合成样本：语义准确率 24/24；MATCH（A）12/12；MISMATCH（B）12/12；ties=0；
  median d_raw(MATCH)=+19.91>0；median d_raw(MISMATCH)=−12.00<0；greedy 一致性 24/24（见 `synthetic_readout_audit.csv`、`greedy_diagnostic.csv`）。

## 3. dev 四格行为结果（195 组，T0/T1/T2）

| 模板 | cell accuracy（OO/OS/SO/SS） | ACC_o | ACC_s | RPAG | FR_SS | FA_SO | tie |
|---|---|---|---|---|---|---|---|
| T0 | {acc_t0['OO']:.3f} / {acc_t0['OS']:.3f} / {acc_t0['SO']:.3f} / {acc_t0['SS']:.3f} | {P2['ACC_o']:.3f} | {P2['ACC_s']:.3f} | {P2['RPAG']:.3f} | {P2['false_reject_SS']:.3f} | {P2['false_accept_SO']:.3f} | {P2['tie_rate']:.3f} |
| T1 | {acc_t1['OO']:.3f} / {acc_t1['OS']:.3f} / {acc_t1['SO']:.3f} / {acc_t1['SS']:.3f} | {P3['template_summaries']['T1']['ACC_o']:.3f} | {P3['template_summaries']['T1']['ACC_s']:.3f} | {P3['template_summaries']['T1']['RPAG']:.3f} | {P3['template_summaries']['T1']['false_reject_SS']:.3f} | {P3['template_summaries']['T1']['false_accept_SO']:.3f} | {P3['template_summaries']['T1']['tie_rate']:.3f} |
| T2 | {acc_t2['OO']:.3f} / {acc_t2['OS']:.3f} / {acc_t2['SO']:.3f} / {acc_t2['SS']:.3f} | {P3['template_summaries']['T2']['ACC_o']:.3f} | {P3['template_summaries']['T2']['ACC_s']:.3f} | {P3['template_summaries']['T2']['RPAG']:.3f} | {P3['template_summaries']['T2']['false_reject_SS']:.3f} | {P3['template_summaries']['T2']['false_accept_SO']:.3f} | {P3['template_summaries']['T2']['tie_rate']:.3f} |

### SS error retention（以 T0 的 {P3['t0_ss_error_groups']} 个 SS 错拒 group 为锚）

```text
SS_error_retention(T1) = {P3['retention']['T1']:.3f}
SS_error_retention(T2) = {P3['retention']['T2']:.3f}
```

### Bootstrap（2,000 次 source-group 重采样，seed=20260811，95% CI；见 `bootstrap_behavior_metrics.csv`）

| 模板 | metric | 95% CI |
|---|---|---|
| T0 | SS_false_rejection_rate | [{P2['bootstrap_FR_SS_ci'][0]:.3f}, {P2['bootstrap_FR_SS_ci'][1]:.3f}] |
| T0 | RPAG | [{P2['bootstrap_RPAG_ci'][0]:.3f}, {P2['bootstrap_RPAG_ci'][1]:.3f}] |
| T1 | SS_false_rejection_rate | [{P3['bootstrap']['T1']['FR_SS_ci'][0]:.3f}, {P3['bootstrap']['T1']['FR_SS_ci'][1]:.3f}] |
| T1 | SS_error_retention | {P3['bootstrap']['T1']['retention']:.3f} |
| T2 | SS_false_rejection_rate | [{P3['bootstrap']['T2']['FR_SS_ci'][0]:.3f}, {P3['bootstrap']['T2']['FR_SS_ci'][1]:.3f}] |
| T2 | SS_error_retention | {P3['bootstrap']['T2']['retention']:.3f} |

## 4. 行为资格门判定

### 4.1 T0 门（继承 D1 原始协议，`d1_gate.py`）

| 条件 | 值 | 通过 |
|---|---|---|
| 1. 读出语义回归全部通过 | 24/24 等 | ✓ |
| 2. ACC_o ≥ 0.85 | {P2['ACC_o']:.3f} | ✓ |
| 3. RPAG ≥ 0.15 | {P2['RPAG']:.3f} | ✓ |
| 4. SS 或 SO：错误组 ≥50 且错误率 ≥0.25 | SS：{P2['ss_error_groups']} 组 / {P2['ss_error_rate']:.3f} | ✓ |
| 5. 四格总 tie_rate ≤ 0.02 | {P2['tie_rate']:.3f} | ✓ |
| 6. 无 NaN/截断/解析失败 | {P2['nan_inf']} | ✓ |

**T0 门通过**：Mistral 在 T0 上存在足量的 reference-adherence failure（SS 错拒 76/195 组，率 0.390）。

### 4.2 T1/T2 模板稳健性门（继承 D1-R 原始协议，`d1r_gate.py`）

| 条件（对 T1/T2 各自） | T1 | T2 |
|---|---|---|
| 1. ACC_o ≥ 0.95 | {P3['template_summaries']['T1']['ACC_o']:.3f} ✓ | {P3['template_summaries']['T2']['ACC_o']:.3f} ✓ |
| 2. SS false_reject ≥ 0.50 | {P3['template_summaries']['T1']['false_reject_SS']:.3f} ✗ | {P3['template_summaries']['T2']['false_reject_SS']:.3f} ✗ |
| 3. FR bootstrap CI lower ≥ 0.40 | {P3['bootstrap']['T1']['FR_SS_ci'][0]:.3f} ✓ | {P3['bootstrap']['T2']['FR_SS_ci'][0]:.3f} ✗ |
| 4. RPAG ≥ 0.20 | {P3['template_summaries']['T1']['RPAG']:.3f} ✓ | {P3['template_summaries']['T2']['RPAG']:.3f} ✓ |
| 5. SS_error_retention ≥ 0.60 | {P3['retention']['T1']:.3f} ✓ | {P3['retention']['T2']:.3f} ✓ |
| 6. 四格总 tie_rate ≤ 0.02 | {P3['template_summaries']['T1']['tie_rate']:.3f} ✓ | {P3['template_summaries']['T2']['tie_rate']:.3f} ✓ |
| 7. 无 NaN/截断/解析失败 | ✓ | ✓ |

**T1/T2 门未通过**（条件 2 对两者均失败；条件 3 对 T2 失败）。
按照 D1-R 可唯一恢复的门槛原样继承，Mistral 的 SS 错拒率在替代表述下不足量（FR_SS 0.48 / 0.46 < 0.50），
不满足"跨候选表述稳健且足量的 reference-adherence failure"。

**最终标签：`mistral_template_robustness_insufficient`**

## 5. 结论边界

- 本轮确认：Mistral 在 T0（D1 原模板）上存在足量 reference-adherence failure（SS 错拒 76 组 / 率 0.390），T0 门通过。
- 本轮否定：Mistral 的该现象在 D1-R 继承的模板稳健性门槛下不成立——替代表述 T1/T2 中 SS 错拒率不足量
  （FR_SS<0.50），即不满足"跨候选表述稳健且足量"的 H1 复现要求。
- 不得声称：Mistral 与 Qwen 具有跨模型一致的 reference-adherence 机制；不得把 Mistral 与 Qwen 结果做 pooled average；
  "参数知识是唯一原因"不成立。
- 后续 Mistral true-prefix representation monitor 本轮未授权（Phase 4 未运行）。
""",
    encoding="utf-8")

# ---------------------------------------------------------------------------
# artifacts/decision.json
# ---------------------------------------------------------------------------
decision = {
    "final_label": final_label,
    "final_reserve_read": False,
    "hidden_states_read": False,
    "probe_trained": False,
    "activation_intervention_run": False,
    "prompt_baselines_run": False,
    "train_text_read": False,
    "model": "mistralai/Mistral-7B-Instruct-v0.3 (local BF16)",
    "phase0_inheritance": "passed (D0 label/dev 195/swap/four-cell/T0 rendering)",
    "phase1_decision_channel": "passed (A->1098, B->1133; synthetic 24/24; ties 0)",
    "phase2_t0_gate": {
        "passed": True,
        "ACC_o": P2["ACC_o"], "RPAG": P2["RPAG"],
        "SS_error_groups": P2["ss_error_groups"], "SS_false_rejection_rate": P2["false_reject_SS"],
        "FR_CI95": P2["bootstrap_FR_SS_ci"], "gate_source": "D1 d1_gate.py (unique recovery)",
    },
    "phase3_template_gate": {
        "passed": False,
        "reason": "D1-R inherited FR_SS>=0.50 unmet for T1/T2; T2 CI lower<0.40",
        "retention": P3["retention"],
        "FR_SS": {"T1": P3["template_summaries"]["T1"]["false_reject_SS"],
                  "T2": P3["template_summaries"]["T2"]["false_reject_SS"]},
        "gate_conditions": P3["gate_conditions"],
        "gate_source": "D1-R d1r_gate.py (unique recovery)",
    },
}
(R / "artifacts" / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
print("wrote failure_examples.md, final_report.md, artifacts/decision.json")
print("FINAL:", final_label)
