# inheritance_audit.md

## 继承来源（只读，未修改任何既有目录）

- D0 `fixed_split_indices.json`：split 索引与哈希（train 587 / dev 195 / reserve 197）。
- D0 `preliminary_swap_pairs.jsonl`：swap 映射（仅流式提取允许 group 的 q/r_o/r_s；不复制文本）。
- D1 `_prompt_constants.json`（system / user 模板 / accept 362 / reject 425）与 `synthetic_pair_manifest.json`（24 pairs）。
- D2-R1：冻结规格（L18/C=0.01/B_surface 9 特征）、train 587 hidden states（`prefix_hidden_states/train_*.npz`）、
  train/dev SS 标签（`_ss_train_scores.json` / `_ss_dev_scores.json`）、prefix 构造与 offset-mapping 规则。
- D4-M0 `leak_isolation_audit.md`：final-reserve 泄露 group 唯一标识。

## 冻结规格确认

| 项 | 值 |
|---|---|
| final label（继承目标） | `true_prefix_reference_state_signal_localized` |
| 冻结 layer | 18 |
| 冻结 token | R_end（Reference Answer 正文最后一个非空白 token） |
| 冻结 classifier | L2 Logistic Regression, C=0.01, max_iter=2000, class_weight=balanced, StandardScaler train-only |
| 标签方向 | y=1 为后续 SS false rejection |
| prefix | 完整输入真实截断至 R_end（含 R_end），其后无 Candidate/Answer:/generation prompt token |

## 未执行项

- 未运行行为资格重跑、Mistral 实验、prompt baseline、activation intervention、新 Probe 搜索。
- 未在 dev/final 上拟合、校准或选择任何超参数。
- 未读取/评分/缓存隔离 group（`0075758e…`）。
