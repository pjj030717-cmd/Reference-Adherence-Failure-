# D3 未执行阶段说明

分段零扰动等价门（协议 §4）失败，标签为 `segmented_execution_equivalence_invalid`。按协议"任何不满足，立即停止"，以下阶段全部未执行：

## 方向构造（协议 §6）

- 未训练 RBSP 方向 probe；
- 未计算 D3-fit / D3-tune 切分；
- 未计算 `v = coef / ||coef||`、`q`、5-fold cosine 诊断；
- 未产生 `rbsp_direction_diagnostics.csv`。

## D3-tune 冻结选择（协议 §7）

- 未运行任何 `(coverage, alpha)` 配置的干预；
- 未计算 `Δd_SS / Δd_OO / Δd_OS / Δd_SO / CSI`、SS hard rescue rate；
- 未产生 `rbsp_config_grid_train_only.csv` 与 `rbsp_config_selection_train_only.json`。

## 必要对照与 dev 因果确认（协议 §8–§9）

- 未运行 `B_zero / B_random / B_reverse / M_RBSP`；
- 未产生 `metrics_by_method_cell_dev.csv`、`paired_bootstrap_causal_effects.csv`、`random_direction_control_audit.csv`、`reverse_direction_audit.csv`。

## 原则

不伪造、不填充任何干预数值。若后续需要在不同实现/环境下复核分段执行等价性，需重新满足 §4 门之后才可继续。
