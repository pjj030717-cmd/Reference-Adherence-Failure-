# frozen_surface_baseline_reconstruction_audit.md

## B_surface 冻结规格（从 D2-R1 analysis.py 唯一恢复）

- 特征（9 个，预注册模型无关表面特征）：
```text
  - q_token_count
  - r_o_token_count
  - r_s_token_count
  - abs(r_o_tokens - r_s_tokens)
  - q_char_count
  - r_s_char_count
  - r_s_word_count
  - has_hyphen
  - is_multiword
```
- r_o 来源：D0 `preliminary_swap_pairs.jsonl` train 行（original correct answer，流式提取 train 行）。
- 分类器：`LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')`（C 由 train 5-fold 组 CV 冻结）。
- 标准化：`StandardScaler().fit(X_surface_train)`（只在 train 拟合）。
- D2-R1 dev 基准：AUROC=0.6207590569292697, AUPRC=0.8181915706506249, C=1.0, features=9。

## 重建方式

- 特征按 D2-R1 `build_surface` 逻辑重建；r_o 从 D0 swap train 行获取（D0 为允许来源，协议 0.3 授权用 D0 train 重建）。
- 重建不读取 dev/final 标签、特征或结果来选择超参数（C 继承冻结值 1.0，train CV 复核）。
