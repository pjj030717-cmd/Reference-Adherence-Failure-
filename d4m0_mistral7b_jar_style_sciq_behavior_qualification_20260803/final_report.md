# Final Report — D4-M0 Mistral-7B-Instruct-v0.3 JAR-style SciQ 行为资格门

| 问题 | 结果 |
|---|---|
| D0 数据与 dev split 是否唯一继承？ | 是（D0 标签 `jar_style_sciq_data_qualification_feasible`；dev=195 组；swap/四格/T0 渲染语义唯一恢复；无 train/final-reserve 文本进入本目录） |
| Mistral 是否可本地 BF16 加载？ | 是（本地路径，BF16/eval/inference_mode/batch=1，revision 本地不可得；见 `model_access_audit.md`） |
| A/B continuation 是否公平且语义正确？ | 是（` A`→id 1098，` B`→id 1133；单 token、不同 id、无 UNK、等长；24/24 synthetic 回归通过） |
| T0 是否通过 reference-adherence 行为门？ | 是（ACC_o=0.995≥0.85，RPAG=0.341≥0.15，SS 错拒 76 组≥50 且率 0.390≥0.25，ties=0，无 NaN） |
| T1/T2 是否通过模板稳健性门？ | 否（D1-R 继承门槛 FR_SS≥0.50 未满足：T1=0.482，T2=0.456；T2 的 CI lower=0.385<0.40） |
| 是否允许进入 Mistral true-prefix Probe？ | 否 |
| final-reserve 是否完全未触碰？ | 否（仅 Phase 0 结构探索 `head -3` 时显示了 1 行 `split=final_reserve` 的题目文本；未评分、未读取标签/缓存/hidden state、未复制入本目录；见第 6 节诚实披露） |
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
| T0 | 0.990 / 1.000 / 0.697 / 0.610 | 0.995 | 0.654 | 0.341 | 0.390 | 0.303 | 0.000 |
| T1 | 0.985 / 0.995 / 0.733 / 0.518 | 0.990 | 0.626 | 0.364 | 0.482 | 0.267 | 0.001 |
| T2 | 0.985 / 1.000 / 0.667 / 0.544 | 0.992 | 0.605 | 0.387 | 0.456 | 0.333 | 0.000 |

### SS error retention（以 T0 的 76 个 SS 错拒 group 为锚）

```text
SS_error_retention(T1) = 0.987
SS_error_retention(T2) = 0.961
```

### Bootstrap（2,000 次 source-group 重采样，seed=20260811，95% CI；见 `bootstrap_behavior_metrics.csv`）

| 模板 | metric | 95% CI |
|---|---|---|
| T0 | SS_false_rejection_rate | [0.318, 0.456] |
| T0 | RPAG | [0.287, 0.397] |
| T1 | SS_false_rejection_rate | [0.410, 0.549] |
| T1 | SS_error_retention | 0.987 |
| T2 | SS_false_rejection_rate | [0.385, 0.528] |
| T2 | SS_error_retention | 0.961 |

## 4. 行为资格门判定

### 4.1 T0 门（继承 D1 原始协议，`d1_gate.py`）

| 条件 | 值 | 通过 |
|---|---|---|
| 1. 读出语义回归全部通过 | 24/24 等 | ✓ |
| 2. ACC_o ≥ 0.85 | 0.995 | ✓ |
| 3. RPAG ≥ 0.15 | 0.341 | ✓ |
| 4. SS 或 SO：错误组 ≥50 且错误率 ≥0.25 | SS：76 组 / 0.390 | ✓ |
| 5. 四格总 tie_rate ≤ 0.02 | 0.000 | ✓ |
| 6. 无 NaN/截断/解析失败 | 0 | ✓ |

**T0 门通过**：Mistral 在 T0 上存在足量的 reference-adherence failure（SS 错拒 76/195 组，率 0.390）。

### 4.2 T1/T2 模板稳健性门（继承 D1-R 原始协议，`d1r_gate.py`）

| 条件（对 T1/T2 各自） | T1 | T2 |
|---|---|---|
| 1. ACC_o ≥ 0.95 | 0.990 ✓ | 0.992 ✓ |
| 2. SS false_reject ≥ 0.50 | 0.482 ✗ | 0.456 ✗ |
| 3. FR bootstrap CI lower ≥ 0.40 | 0.410 ✓ | 0.385 ✗ |
| 4. RPAG ≥ 0.20 | 0.364 ✓ | 0.387 ✓ |
| 5. SS_error_retention ≥ 0.60 | 0.987 ✓ | 0.961 ✓ |
| 6. 四格总 tie_rate ≤ 0.02 | 0.001 ✓ | 0.000 ✓ |
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

## 6. 合规性诚实披露

- 在 Phase 0 探索 D0 文件结构时，曾执行 `head -3` 查看 D0 的 `eligible_source_groups.jsonl` 与
  `preliminary_swap_pairs.jsonl` 各前 3 行，其中个别行包含 train 组的题目文本，且
  `preliminary_swap_pairs.jsonl` 有一行 `split=final_reserve` 的题目文本被显示。
- 该等文本**未被复制到本目录**、未用于任何评分、特征提取、标签或样本选择；本目录内所有正式
  dev 数据均只经 D1 `scripts/_dev_pairs.jsonl`（dev-only，195 组）流式读取；未读取任何
  final-reserve 评分、标签、缓存或 hidden state。
- 为保持诚实，`artifacts/decision.json` 中 `train_text_read` 与 `final_reserve_read` 如实标记为 `true`
  并附此说明；科学结论与最终标签不受影响（该读取仅发生在结构理解阶段，且无任何数据流进入分析）。
- 其余禁令均确认未违反：零 hidden state、零 Probe、零干预、零 prompt baseline。

## 7. 泄露隔离与审计（2026-08-03 补充）

- 已对泄露事件完成隔离审计（见 `leak_isolation_audit.md`）：
  - **id 级隔离**：本目录所有文件中的 64-hex D0 group id 全部 ∈ dev 195；train/final_reserve group id 零命中。
  - **文本级隔离**：已泄露的 train / final-reserve 组题目文本在本目录任何文件中零命中（指纹比对由目录外脚本执行，本目录不存放泄露内容）。
  - **正包含**：`t0_metrics_by_cell_dev.csv`、`t1_t2_metrics_by_cell_dev.csv` 的唯一 id 集合恰等于 dev 195；`template_error_retention_audit.csv` 76 个 id ⊆ dev；`failure_examples.md` 仅含 dev 前缀。
- **隔离结论：PASS**——泄露内容未进入任何文件，未流入任何指标/门判断/结论；最终标签 `mistral_template_robustness_insufficient` 与全部科学结论不受影响。



