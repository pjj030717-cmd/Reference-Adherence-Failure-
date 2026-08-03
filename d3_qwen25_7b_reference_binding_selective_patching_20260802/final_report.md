# final_report.md

| 问题 | 结果 |
|---|---|
| D0–D2-R1 是否被唯一继承？ | 是（五标签 + 模型哈希 + 模板哈希全部核验通过） |
| segmented-zero 是否精确复现 D1？ | **否**（3/780 标签翻转；d_raw 中位差 0.125、max 2.125；max BF16-ULP 偏差 21798） |
| hook 是否修改 D2-R1 的同一层/同一 R_end state？ | 未执行（分段等价门失败，未进入 hook 干预阶段） |
| 所有方向、幅度、coverage 是否仅在 train 内冻结？ | 未执行（同上） |
| RBSP 是否改善 SS？ | 未执行（同上） |
| OO/OS/SO 是否保持稳定？ | 未执行（同上） |
| RBSP 是否优于随机等范数方向？ | 未执行（同上） |
| 是否读取/评分 final-reserve？ | 否 |
| 是否允许进入最终确认与外部泛化？ | 否 |
| 最终标签 | `segmented_execution_equivalence_invalid` |

---

## 1. 唯一继承

D3 Phase 0（`scripts/d3_phase0_inheritance.py`）逐项核验：

```text
D0   == jar_style_sciq_data_qualification_feasible      ✓
D1   == jar_style_reference_override_behavior_feasible  ✓
D1-R == template_robust_reference_override_feasible     ✓
D2   == prefix_causality_audit_invalid                  ✓
D2-R1 == true_prefix_reference_state_signal_localized   ✓
```

- 模型 `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c70…`；config / tokenizer / safetensors index 哈希与 D1 一致。
- D0 固定 split seed `20260802`：train 587 / dev 195 / final-reserve 197。
- 本轮只使用 train + dev；final-reserve 197 groups 未读取、未评分、未缓存、未统计。
- 基础模板 T0 = `The answer is <answer>.`，A/B continuation `" A"` / `" B"`（id 362 / 425），teacher-forced pos = prompt_len - 1。
- D2-R1 选定表示层 `hidden-state index 18`、`C = 0.01`、token `true-prefix R_end`，作为 RBSP 继承目标。

## 2. 固定行为任务

四格保持 D1 定义（OO/OS/SO/SS 期望 A/B/B/A），`d_raw = log p(" A") - log p(" B")` 读出不变。本轮未更改 prompt、模板、swap、label、阈值、切分或 continuation。

## 3. 核心执行方式：真 prefix → cache → suffix

按协议实现两阶段执行：

- 阶段 P：从实际 chat-rendered token ids 截断到 `R_end`，`use_cache = true`；
- 阶段 S：继承 `past_key_values`，用 `cache_position = arange(prefix_len, prefix_len+suffix_len)` 继续处理，直至 `prompt_len - 1` 读取 A/B logits。

实现文件：`scripts/d3_core.py`。规格详见 `segmented_execution_spec.md`。

## 4. 分段执行的零扰动等价门（决定性结果）

对 dev 780 个 T0 输入运行 `segmented-zero`（alpha=0，不修改任何激活），与 D1 monolithic 比较：

| 指标 | 要求 | 实测 |
|---|---|---|
| predicted labels | 780/780 一致 | **777/780（3 翻转）** |
| d_raw BF16 序列化精度 | 一致 | **不一致（mean 0.183 / max 2.125）** |
| l_A / l_B | BF16 序列化精度 | **max ULP 偏差 21798** |
| OO / OS / SO / SS | 1.000 / 1.000 / 0.928 / 0.241 | seg 端 1.0000 / 1.0000 / 0.9897* / 0.9949* |

（*SO/SS 列为 segmented 端与 D1 标签的一致率，非 D1 原始 accuracy。）

mismatch 明细（`segmented_zero_equivalence_audit.csv`）：

```text
53a9a275a436 SO  D1=B → seg=A  Δd=+1.4375
9a48b8a8d35c SO  D1=B → seg=A  Δd=+0.7500
bb16e19b3e8a SS  D1=A → seg=B  Δd=-0.6875
```

### 根因论证（多路交叉验证）

1. **D1 参考可信**：3 个 mismatch 样本用完整前向重算，与 D1 CSV 逐位一致（|Δd|=0.0）。
2. **续算 API 正确**：toy 24-token 序列在 split=12 时，分段续算与完整前向 diff = 0.000000（数学等价）；split 未对齐时约 1 ULP kernel 噪声。
3. **模型对序列总长度固有数值敏感**：即使 eager + 严格 causal mask，pos0（只 attend 自己）hidden state 随序列长度变化（layer1 0.0078 → layer28 23.56）；phase P 前缀 KV 与 monolithic 前缀 KV 数值不同（key 2.0 / value 0.97）；D2-R1 已记录同一现象（eager 与 sdpa 均复现）。
4. 结论：真截断 prefix 前向（99 tokens）与完整序列前向（116 tokens）对同一 `R_end` 位置的数值必然不同，且经 28 层放大后足以翻转边界判决。

按协议 §4 与 §11，不得换 backend / batch / 精度 / padding / cache 实现规避。因此立即停止，判定 `segmented_execution_equivalence_invalid`。

## 5. L18 与 hook 映射审计

**未执行**。协议顺序要求先通过分段零扰动等价门（§4），该门失败即停止，不进入 hook 干预（§5）。`hook_layer_mapping_audit.md` 说明见下。

## 6. RBSP 方向构造 / 7. D3-tune 冻结选择 / 8. 必要对照 / 9. dev 因果确认

**全部未执行**。`rbsp_direction_diagnostics.csv`、`rbsp_config_grid_train_only.csv`、`rbsp_config_selection_train_only.json`、`metrics_by_method_cell_dev.csv`、`paired_bootstrap_causal_effects.csv`、`random_direction_control_audit.csv`、`reverse_direction_audit.csv` 均因等价门失败而未产生（见各未执行说明文件）。

## 10. 决定门

分段零扰动等价门（§4 条件 1）失败 → 无法满足决定门，输出 `segmented_execution_equivalence_invalid`。

本结论只否定"该实现下当前分段执行方案 + 干预实验不可行"；不否定 D1/D1-R 行为现象，也不否定 D2-R1 pre-decision readout。RBSP 的因果问题（L18/R_end 方向对 SS 的选择性因果作用）在本环境、本实现下**无法通过分段执行这一必要前提进行因果确认**。

## 11. 严格禁止复核

```text
final_reserve 读取/评分/缓存      ：未发生
换 backend/batch/精度/padding     ：未发生（保持 sdpa 与 D1 一致）
新增 alpha/coverage/方向/随机基线 ：未发生
LoRA/SFT/RM 训练                 ：未发生
prompt baseline / verifier / Mistral：未发生
写回 D0/D1/D1-R/D2/D2-R1         ：未发生
```

## 12. 交付物

见目录清单。`artifacts/decision.json` 已写入最终标签。

## 未执行阶段说明

以下文件因 §4 等价门失败而未执行（本文件如实声明，不伪造）：

- `hook_layer_mapping_audit.md`
- `rbsp_direction_diagnostics.csv`
- `rbsp_config_grid_train_only.csv`
- `rbsp_config_selection_train_only.json`
- `metrics_by_method_cell_dev.csv`
- `paired_bootstrap_causal_effects.csv`
- `random_direction_control_audit.csv`
- `reverse_direction_audit.csv`

协议要求这些文件在通过 §4 门之后产生；门失败即停止，不产生任何干预相关数值。
