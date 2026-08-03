# Final Report — E01-D1-R 候选表述稳健性资格门（Qwen2.5-7B-Instruct × SciQ）

| 问题 | 结果 |
|---|---|
| D0/D1 是否被唯一继承？ | 是（D0 标签/D1 标签/dev 195 组/模型哈希/prompt/continuation 全部一致） |
| D0 hash 缺陷是否已透明记录且不影响模板内容？ | 是（见 `provenance_amendment.md`；模板字符串+UTF-8 SHA256 可唯一恢复） |
| T0 是否逐行复现 D1？ | 是（780/780 行预测与 l_A/l_B/d_raw 一致；OO=1.000, OS=1.000, SO=0.928, SS=0.241；ACC_o=1.000, ACC_s=0.585, RPAG=0.415） |
| T1 是否保留 SS reference-binding failure？ | 是（FR_SS=0.831；retention=1.000；RPAG=0.436） |
| T2 是否保留 SS reference-binding failure？ | 是（FR_SS=0.887；retention=0.993；RPAG=0.444） |
| 是否读取/评分 final-reserve？ | 否（final_reserve_model_scored=false） |
| 是否读取 hidden states 或训练 Probe？ | 否（hidden_states_read=false, probe_trained=false） |
| 是否允许进入 D2 表示机制定位？ | 是 |
| 最终标签 | `template_robust_reference_override_feasible` |

## 1. 继承（Phase 0）

- D0：`jar_style_sciq_data_qualification_feasible`；dev=195（seed 20260802）。
- D1：`jar_style_reference_override_behavior_feasible`；synthetic readout 24/24。
- 模型：Qwen2.5-7B-Instruct，revision `a09a3545…`；config/tokenizer/index 哈希与 D1 一致。
- Prompt：system/user 模板/chat template/continuations `" A"`/`" B"`/teacher-forced pos=prompt_len-1/BF16/eval/inference_mode/batch=1 全部逐字继承。
- 基础模板 T0：`The answer is <answer>.`（SHA256 `c42e1ea1…`）。

## 2. D0 hash 缺陷修正说明

见 `provenance_amendment.md`。`candidate_rendering_spec.json` 的 `sha256_utf8` 字段为中间态哈希，
与最终文件全文不一致；但模板字符串本身及 979 对渲染均正确，不影响 D1-R 继承。

## 3. T0 精确复现

- 195 group × 4 cell = 780 行，全部与 D1 `four_cell_scores_dev.csv` 逐行对齐：
  - 预测标签 780/780 完全一致；
  - l_A / l_B / d_raw 在 BF16 精度下一致（|Δ| ≤ 1e-3）；
  - aggregate：OO=1.000, OS=1.000, SO=0.928, SS=0.241；ACC_o=1.000, ACC_s=0.585, RPAG=0.415。

## 4. T1/T2 行为结果

| 模板 | cell accuracy（OO/OS/SO/SS） | ACC_o | ACC_s | RPAG | FR_SS | FA_SO | tie |
|---|---|---|---|---|---|---|---|
| T0 | 1.000 / 1.000 / 0.928 / 0.241 | 1.000 | 0.585 | 0.415 | 0.759 | 0.072 | 0.000 |
| T1 | 0.995 / 1.000 / 0.954 / 0.169 | 0.997 | 0.562 | 0.436 | 0.831 | 0.046 | 0.000 |
| T2 | 0.964 / 1.000 / 0.964 / 0.113 | 0.982 | 0.538 | 0.444 | 0.887 | 0.036 | 0.000 |

### SS error retention（以 T0 的 148 个 SS 错拒 group 为锚）

```text
SS_error_retention(T1) = 148/148 = 1.000
SS_error_retention(T2) = 147/148 = 0.993
```

### Bootstrap（1,000 次 source-group 重采样，95% CI）

| 模板 | metric | 95% CI |
|---|---|---|
| T1 | false_reject_SS | [0.774, 0.882] |
| T1 | RPAG | [0.405, 0.464] |
| T1 | SS_error_retention | [1.000, 1.000] |
| T2 | false_reject_SS | [0.841, 0.928] |
| T2 | RPAG | [0.413, 0.472] |
| T2 | SS_error_retention | [0.989, 1.000] |

## 5. 资格门判定

| 条件（对 T1 和 T2 各自） | T1 | T2 |
|---|---|---|
| 1. ACC_o ≥ 0.95 | 0.997 ✓ | 0.982 ✓ |
| 2. SS false_reject ≥ 0.50 | 0.831 ✓ | 0.887 ✓ |
| 3. FR bootstrap CI lower ≥ 0.40 | 0.774 ✓ | 0.841 ✓ |
| 4. RPAG ≥ 0.20 | 0.436 ✓ | 0.444 ✓ |
| 5. SS_error_retention ≥ 0.60 | 1.000 ✓ | 0.993 ✓ |
| 6. 四格总 tie_rate ≤ 0.02 | 0.000 ✓ | 0.000 ✓ |
| 7. 无 NaN/截断/解析失败 | ✓ | ✓ |

**最终标签：`template_robust_reference_override_feasible`**

## 6. 结论边界

- SS reference-binding failure 在三种独立候选表述（T0/T1/T2）下持续存在，方向一致（SS 错拒为主），
  且 T0 的错拒 group 在 T1/T2 中几乎全部保持（retention ≥ 0.99）。
- 本轮仅确认现象对候选表述稳健，不构成"参数知识是唯一原因"的因果证据；
- 允许进入 D2 表示机制定位作为后续独立门的前提，本轮未执行任何机制实验。
