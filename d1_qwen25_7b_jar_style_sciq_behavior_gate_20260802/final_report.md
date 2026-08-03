# Final Report — E01-D1 Qwen2.5-7B-Instruct JAR-style SciQ 行为资格门

| 问题 | 结果 |
|---|---|
| D0 是否被唯一、完整地继承？ | 是（标签/模板/split seed/split hashes 重算一致；dev=195 组） |
| Qwen2.5-7B 是否本地可访问？ | 是（revision `a09a3545…`，BF16/eval/inference_mode，batch=1） |
| A/B continuation 是否公平且可评分？ | 是（` A`→id 362，` B`→id 425；单 token、长度相同、无 UNK） |
| 正确 teacher-forced 语义回归是否通过？ | 是（24/24 准确率；A/B 各 12/12；ties=0；greedy 一致 24/24） |
| 原始参考行为 `ACC_o` 是否通过？ | 是（ACC_o = 1.000 ≥ 0.85） |
| 替换参考下是否存在足量参考服从错误？ | 是（SS false-reject 错误 group 148 ≥ 50，错误率 0.759 ≥ 0.25；RPAG=0.415） |
| 是否读取或评分 final-reserve？ | 否（final_reserve_model_scored = false） |
| 是否读取 hidden states / 训练 Probe？ | 否（hidden_states_read=false, probe_trained=false） |
| 是否允许进入机制表示实验？ | 是 |
| 最终标签 | `jar_style_reference_override_behavior_feasible` |

## 1. 继承与模型

- D0 继承：`jar_style_sciq_data_qualification_feasible`；模板 SHA256（模板字符串）`c42e1ea1…`；
  split seed 20260802；train/dev/final_reserve 各 split SHA256 重算一致（见 `inheritance_audit.md`）。
- D0 缺陷披露：`candidate_rendering_spec.json` 的 `sha256_utf8` 字段为中间态哈希（D0 先写 null 再回填导致），
  与最终文件全文哈希不同；模板字符串本身可唯一恢复且 979 对渲染全部符合模板，不影响本门。
- 模型：Qwen2.5-7B-Instruct，本地路径 `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`，revision `a09a3545…`；
  BF16、`model.eval()`、`torch.inference_mode()`、`batch_size=1`（见 `model_access_audit.md`）。

## 2. Tokenizer 与读出语义回归

- continuation：` A`→id 362，` B`→id 425；两者均单 token、continuation length 相同、token id 不同、无 UNK（见 `tokenization_audit.md`）。
- 24 条合成样本（12 一致/12 不一致，无歧义、无 D0/SciQ 文本）：
  - 语义准确率 24/24（≥ 23/24）；A 类 12/12、B 类 12/12（各 ≥ 11/12）
  - ties = 0；median d_raw(A) = +20.72 > 0；median d_raw(B) = −22.94 < 0
  - greedy 一致性 24/24（见 `synthetic_readout_audit.csv`）
- 空白 prompt 诊断：d_raw = −9.34（仅记录，未用于校正，非失败条件）。

## 3. dev 四格行为结果（195 组）

| cell | n | accuracy | accept_rate | mean d_raw | median d_raw | tie_rate |
|---|---:|---:|---:|---:|---:|---:|
| OO (`r_o`/`c_o`) | 195 | 1.000 | 1.000 | +16.38 | +16.74 | 0.000 |
| OS (`r_o`/`c_s`) | 195 | 1.000 | 0.000 | −21.06 | −21.28 | 0.000 |
| SO (`r_s`/`c_o`) | 195 | 0.928 | 0.072 | −14.25 | −16.50 | 0.000 |
| SS (`r_s`/`c_s`) | 195 | 0.241 | 0.241 | −8.13 | −12.91 | 0.000 |

```text
ACC_o  = 1.0000
ACC_s  = 0.5846
RPAG   = 0.4154
false_reject_SS = 0.7590
false_accept_SO = 0.0718
override_error_group 数 = 151（SO 或 SS 至少一处错误）
SS 错误 group = 148（错误率 0.759）；SO 错误 group = 14（错误率 0.072）
```

### Bootstrap（1,000 次 source-group 重采样，95% CI）

| metric | 95% CI |
|---|---|
| ACC_o | [1.0000, 1.0000] |
| ACC_s | [0.5513, 0.6205] |
| RPAG | [0.3795, 0.4487] |
| false_reject_SS | [0.7026, 0.8154] |
| false_accept_SO | [0.0410, 0.1077] |

## 4. 参数知识诊断（描述性，不参与样本选择）

- 无 reference 贪心短答：`knowledge_matches_original` = 90/195 = 0.462。
- 分层（仅报告）：
  - km=True（90 组）：SS 错误率 0.778，SO 错误率 0.089
  - km=False（105 组）：SS 错误率 0.743，SO 错误率 0.057
- 说明：SS 错拒在 km=True / km=False 组中均高（≈0.75–0.78），不因是否匹配参数知识而有显著差异；
  该变量仅用于描述，不筛选样本、不改四格标签。

## 5. 行为资格门判定

| 条件 | 值 | 通过 |
|---|---|---|
| 1. 读出语义回归全部通过 | 24/24 等 | ✓ |
| 2. ACC_o ≥ 0.85 | 1.000 | ✓ |
| 3. RPAG ≥ 0.15 | 0.415 | ✓ |
| 4. SS 错拒或 SO 错接：错误 group ≥50 且错误率 ≥0.25 | SS：148 组 / 0.759 | ✓ |
| 5. 四格总 tie_rate ≤ 0.02 | 0.000 | ✓ |
| 6. 无 NaN/截断/解析失败 | 是 | ✓ |

**最终标签：`jar_style_reference_override_behavior_feasible`**

## 6. 结论边界

- 本结果仅确认可复现的 reference-adherence failure（替换参考下 SS 高错拒），
  不证明"参数知识是唯一原因"；不得据此自动断言机制归属。
- 允许进入机制表示实验（hidden-state / Probe / intervention）仅作为后续独立门的前提，本轮未执行任何此类实验。
