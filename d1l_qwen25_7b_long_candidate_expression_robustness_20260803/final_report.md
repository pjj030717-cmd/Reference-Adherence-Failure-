# Final Report — E01-D1-L Qwen2.5-7B-Instruct 长 Candidate 表达与答案位置稳健性行为资格门

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
| 最终标签 | long_candidate_expression_robust |

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

**最终标签：`long_candidate_expression_robust`**

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
