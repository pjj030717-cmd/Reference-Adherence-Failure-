# final_report.md (D3-M)

| 问题 | 结果 |
|---|---|
| D1 行为失效是否被复现？ | 是（Phase 0B：OO/OS/SO/SS 完整前向与 D1 逐位一致，SS 失效保留） |
| 完整前向 zero 是否等价？ | 是（被动 hook 零干预 780/780 一致，max Δd_raw=0.0） |
| L18 hook 映射是否唯一？ | 是（`hidden_states[18]` = `layers[17]` 输出，R_end 逐位一致） |
| 完整输入方向是否在 tune 有信号？ | **未评估**（容量门失败，未拟合方向 Probe） |
| 是否冻结唯一 patch 配置？ | 否（容量门失败，未进入 Phase 2） |
| SS 是否改善？ | 未评估（未做干预） |
| 非 SS 是否受控？ | 未评估 |
| 是否优于 reverse / random controls？ | 未评估 |
| 是否读取 final-reserve？ | 否 |
| 最终标签 | `monolithic_direction_label_capacity_insufficient` |

---

## 1. 唯一继承（Phase 0A）✓

六标签全部核验通过：

```text
D0   == jar_style_sciq_data_qualification_feasible        ✓
D1   == jar_style_reference_override_behavior_feasible    ✓
D1-R == template_robust_reference_override_feasible       ✓
D2   == prefix_causality_audit_invalid（保持原样）        ✓
D2-R1 == true_prefix_reference_state_signal_localized     ✓
D3   == segmented_execution_equivalence_invalid（保持原样）✓
```

- 模型 revision `a09a3545…`；config/tokenizer/safetensors index 哈希与 D1 一致。
- T0 模板 SHA256 = `c42e1ea10a6be…`；continuation `" A"`/`" B"`（362/425）。
- synthetic 24-pair readout：order_accuracy = **24/24**，ties = **0**。

## 2. 完整前向零干预等价（Phase 0B + 0D）✓

- **Phase 0B**：dev 195×4=780 完整前向与 D1 逐位复现：
  - predicted label 780/780 一致；
  - max BF16-ULP 偏差 = **0.0**；max |Δd_raw| = **0.0**；无 NaN、无 tie。
- **Phase 0D**：安装不修改输出的 intervention hook（apply_fn=None）后 780/780 一致，max Δd_raw = 0.0。
- 结论：完整 monolithic 前向（含被动 hook）与 D1 精确等价。

## 3. L18 hook 层映射（Phase 0C）✓

- `hidden_states[18]` 唯一映射到 `model.model.layers[17]`（decoder block 17）的 forward 输出（纯 tensor）。
- 10 条样本（5 dev + 5 train SS）在 R_end 处 `hidden_states[18]` 与 hook 捕获逐位一致（max_diff = 0.0）。

## 4. 方向标签容量（Phase 1）✗ —— 停止

按协议固定程序切分（SHA256 升序 → `random.Random(20260804)` shuffle → 70/30）：

| subset | n | y=1 | y=0 | 门槛 |
|---|---|---|---|---|
| D3M-fit | 411 | 331 | 80 | y1≥80, y0≥40 ✓ |
| D3M-tune | 176 | 137 | **39** | y1≥80, y0≥40 ✗ |

- 标签由完整 T0 monolithic 前向重评分得到，与 D2-R1 train SS 表 587/587 一致。
- D3M-tune y0=39 < 40 → 按协议立即停止：`monolithic_direction_label_capacity_insufficient`。
- 未拟合 Probe、未构造方向 v、未进入方向资格门与 Phase 2/3。

## 5–6. 未执行阶段

- Phase 2（tune grid、冻结 patch 配置）：未执行。
- Phase 3（dev 因果确认、random/reverse controls、bootstrap）：未执行。
- 对应交付物（`frozen_patch_config.json`、`tune_grid_metrics.csv`、`dev_intervention_metrics_by_cell.csv`、`dev_group_level_intervention_audit.csv`、`random_direction_control_metrics.csv`、`bootstrap_causal_effects.csv`）未产生，如实声明。

## 严格禁止复核

```text
final-reserve 读取/评分/缓存        ：未发生
改模型/revision/BF16/batch/chat     ：未发生
改 T0/A-B continuation/teacher-forced：未发生
T1/T2/prompt 工程/SFT/外部 Judge/SAE：未发生
复用 D2/D2-R1 hidden arrays        ：未发生（只核对行为标签 587/587）
换 backend/padding/精度/cache      ：未发生
```

## 本轮边界

```text
D3 的 segmented_execution_equivalence_invalid 保持原样；
本轮不复用 D2 或 D2-R1 的 hidden-state 数组；
本轮不用 prefix cache、past_key_values、分段续算、截断后续算；
本轮只做完整原始输入的 monolithic forward。
```

---

## 结尾声明

本轮若成功，只支持"完整前向下 L18 × R_end 的单方向 patch 对 Judge verdict 有选择性因果影响"。
本轮不证明参数知识是唯一原因，不证明可泛化到其他模型、任务或模板，也不构成最终修复结论。

（注：本轮因方向标签容量门失败而停止，未产生任何干预证据；上述声明按协议要求完整保留。）
