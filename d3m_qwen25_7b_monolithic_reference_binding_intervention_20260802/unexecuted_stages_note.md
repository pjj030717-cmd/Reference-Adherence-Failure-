# D3-M 未执行阶段说明

方向标签容量门（Phase 1）失败，最终标签 `monolithic_direction_label_capacity_insufficient`。
按协议"立即停止"，以下阶段未执行，对应交付文件如实声明未产生：

| 文件 | 阶段 | 状态 |
|---|---|---|
| `full_context_direction_tune_metrics.csv` | Phase 1 tune 方向资格（AUROC / permutation null） | 未执行（容量门失败，未拟合 Probe） |
| `frozen_patch_config.json` | Phase 2 冻结配置 | 未执行 |
| `frozen_patch_config.sha256` | Phase 2 冻结配置哈希 | 未执行 |
| `tune_grid_metrics.csv` | Phase 2 tune grid（coverage × alpha） | 未执行 |
| `dev_intervention_metrics_by_cell.csv` | Phase 3 dev 每方法每格指标 | 未执行 |
| `dev_group_level_intervention_audit.csv` | Phase 3 dev 逐组结果 | 未执行 |
| `random_direction_control_metrics.csv` | Phase 3 10 随机方向对照 | 未执行 |
| `bootstrap_causal_effects.csv` | Phase 3 2000 次 paired bootstrap | 未执行 |

未产生任何干预数值（无 patch、无方向 v、无 alpha/coverage、无随机方向、无 bootstrap）。

## 已产生且已审计的中间产物

- `train_ss_fullforward.json`：587 个 train SS 的完整前向重评分（l_A/l_B/d_raw/pred/y），与 D2-R1 587/587 一致。
- `train_ss_gids.json` / `train_ss_l18_rend.npy`：容量门失败前的特征提取中间产物（train 专用，未参与任何拟合；不涉及 final-reserve）。
- `train_tune_split_manifest.json`：587 group 的切分与 SHA256 清单（独立复核一致）。
