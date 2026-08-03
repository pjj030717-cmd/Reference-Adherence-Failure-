# true_prefix_direction_input_spec.md — 真实 prefix 风险方向输入规格

## 目的

方向构造只用真实截断 reference-prefix 的 L18/R_end hidden state，与 D2-R1 的
prefix 定义完全一致，确保方向反映"Reference 读取时"的已存在风险信号，而非
后续 Candidate 处理引入的信息。

## 1. Prefix 构造（继承 D2-R1）

```text
prefix_input_ids = full_input_ids[:R_end + 1]
prefix_attention_mask = ones_like(prefix_input_ids)
```

- `R_end`：offset mapping 定位 `Reference Answer` 正文最后一个非空白 token。
- prefix 中不存在 Candidate Answer token、`Answer:` token、generation prompt token。
- 将截断序列单独送入模型（batch=1, BF16, eval, inference_mode），
  读取 `hidden_states[18]`（即 L18）最后一位：
  `h_prefix = hidden_states[18][0, prefix_len-1, :]`。

## 2. 本轮重提取与核验

- 本轮对 train 587 组独立重提取 `h_prefix`（float32）。
- 与 D2-R1 存储的 `prefix_hidden_states/train_{group_id}.npz` 中
  `h_prefix[17]`（该数组堆叠 hidden_states[1..28]，故 17 对应 L18）对比：
  - mismatch 数 = 0；
  - max_abs_diff = 0.0（逐位一致）。

## 3. 标签来源

每组标签取自完整 T0-SS 单体前向的 Judge 行为：

```text
y = 1  iff Judge 输出 B（拒绝 reference 对齐候选，SS 错拒）
y = 0  iff Judge 输出 A（接受，SS 接受）
```

本轮重评分并与 D3-M / D2-R1 交叉核对：

- 与 D3-M train_ss_fullforward.json 一致：587/587；
- 与 D2-R1 _ss_train_scores.json 一致：587/587。

## 4. 容量

- train 全部 587 组（y1=468, y0=119），无 fit/tune 切分。
- 标签容量审计见 `train_label_capacity_audit.csv`。

## 5. 产物

- `train_prefix_l18_rend.npy`：shape (587, 3584)，float16（存档）。
- `train_prefix_labels.npy`：shape (587,)，int64。
- `train_prefix_gids.json` / `train_ss_draw.json`。
- 清单见 `true_prefix_hidden_manifest.json`。
