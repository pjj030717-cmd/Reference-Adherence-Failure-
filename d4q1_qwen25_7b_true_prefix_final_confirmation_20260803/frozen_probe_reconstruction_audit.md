# frozen_probe_reconstruction_audit.md

## D2-R1 冻结规格审计（Phase 0.2）

| 项 | 值 | 通过 |
|---|---|---|
| D2-R1 final label | true_prefix_reference_state_signal_localized | ✓ |
| frozen layer | 18 | ✓ |
| frozen C | 0.01 | ✓ |
| dev AUROC (recorded) | 0.9138872915468661 | ✓ |
| classifier config | {"solver": "lbfgs (sklearn default)", "max_iter": 2000, "class_weight": "balanced", "random_state": null, "penalty": "L2", "scaler": "StandardScaler fit on train only", "selection": "layer then C via 5-fold StratifiedGroupKFold(shuffle, random_state=20260802) on TRAIN only"} | ✓ |
| prefix 构造规格 | prefix = full_ids[:R_end+1]; h_prefix 在截断序列末位读取 | ✓ |
| model revision | Qwen/Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28 | ✓ |
| T0 template sha | c42e1ea10a6be334 | ✓ |
| R_end 定位规则 | offset mapping 定位 Reference Answer 正文最后一个非空白 token（见 contract audit 头） | ✓ |
| B_surface dev AUROC | 0.6207590569292697 | ✓ |
| B_surface selected C | 1.0 | ✓ |
| B_surface feature count | 9 | ✓ |
| B_surface 规格 | {"features": ["q_token_count", "r_o_token_count", "r_s_token_count", "abs(r_o_tokens - r_s_tokens)", "q_char_count", "r_s_char_count", "r_s_word_count", "has_hyphen", "is_multiword"], "n_features": 9, "classifier": "LogisticRegression L2, max_iter=2000, class_weight=balanced", "scaler": "StandardScaler fit on train only", "C": "selected via 5-fold group CV on TRAIN only (frozen at 1.0 in D2-R1)", "r_o_source": "D0 swap pairs train rows (original correct answer)"} | ✓ |

## Probe（M_rep）重建规格

- 模型：`Qwen/Qwen2.5-7B-Instruct` revision `a09a3545…`；BF16、eval、inference_mode、batch_size=1。
- 特征：`hidden_states[18][0, prefix_len-1, :]`（L18 × R_end），取自真截断 prefix 单独前向。
- 标准化：`StandardScaler().fit(X_train)`（只在 train 拟合）。
- 分类器：`LogisticRegression(C=0.01, max_iter=2000, class_weight='balanced')`（L2，默认 lbfgs），train 拟合。
- 标签：`y = 1 iff SS predicted_label == 'B'`（后续 SS false rejection）；y=0 iff Accept。
- 层/C 选择：仅 train 5-fold StratifiedGroupKFold CV（random_state=20260802），冻结 18 / 0.01。
- 本目录中无 D2-R1 序列化模型文件（仅 decision.json + 训练脚本）；按协议 0.3 优先顺序第 2 条用 train 587 组重建。

## 特征来源

- train 587 组 hidden states：D2-R1 `prefix_hidden_states/train_*.npz`（D2-R1 自提取，`d2_hidden_arrays_reused=false`）。
- train SS 标签：D2-R1 `scripts/_ss_train_scores.json`（587 行，含 predicted_label）。
- 重建过程不读取 dev/final 标签、特征或结果来选择任何超参数。


## 重建与 dev 只读复现（Phase 0.3，2026-08-03）

- M_rep 重建（train-only）：layer 18, C=0.01, StandardScaler train, LogReg max_iter=2000 balanced。
- B_surface 重建（train-only）：C=1.0（D2-R1 冻结），9 特征，r_o 取自 D0 swap train 行。
- dev 只读复现（不选任何超参数）：

| 模型 | 复现 AUROC | 记录 AUROC | 复现 AUPRC | 记录 AUPRC | 容差 |
|---|---|---|---|---|---|
| M_rep | 0.913887 | 0.9138872915468661 | 0.963219 | 0.9632189056523878 | 0.0001 |
| B_surface | 0.620759 | 0.6207590569292697 | 0.818192 | 0.8181915706506249 | 0.0001 |

- 结论：重建模型在 dev 上与 D2-R1 记录逐行（聚合）一致，冻结 Probe 与 B_surface 唯一恢复完成。
