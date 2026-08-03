# failure_examples.md — D3-M-R1 干预无效的机制诊断

## 结论先行

Development 阶段没有任何 (coverage q, alpha) 配置合格，最终标签为
`monolithic_patch_dev_selectivity_insufficient`。方向可以**预测** SS 错拒
（OOF AUROC 0.922），但在完整单体前向中，L18/R_end 的单点加性干预对该
Judge 最终 logits 的**因果作用**弱到无法翻转任何预测。

## 1. 干预机制已逐层验证（排除实现 bug）

### 1.1 passive hook 零干预等价（Phase 0）

dev 780 行，安装只读 hook，`apply_fn=None`：

```text
label_mismatch = 0
max_abs_delta_d_raw = 0.0
r_end_mismatch = 0
```

### 1.2 hook 确实写入了目标位置

对 dev group `0a10ac...`（SS 输入，d_raw_base=-3.0），在 layers[17] 输出
R_end 位置注入 `delta = 100 * sigma_z * v*`（|delta|=280.35）：

```text
pre  L18/R_end norm = 71.98
post L18/R_end norm = 291.50
diff norm           = 280.34   (≈ |delta|，确认写入成功)
```

### 1.3 修改真实传导到后续层

将 L18/R_end 置零后，重新前向并读 hidden_states：

```text
hidden_states[18][R_end] norm = 0.0      (置零生效)
hidden_states[19][R_end] norm = 102.84   (后续层重新产生了状态)
final d_raw = -2.0625 (vs base -3.0，干预确实改变了输出)
```

### 1.4 阳性对照：同机制在最后一层能翻转

同样 hook 机制加在最后一层（layers[27]）最后一个位置 `+10`：

```text
d_raw: -3.0 → +0.156 (预测翻转)
```

证明 hook 返回值确实被模型采用；L18 干预效应弱不是 hook 失效。

## 2. 弱效应的量化证据

### 2.1 alpha 扫描（dev SS 输入）

| alpha | delta_norm | d_raw | 是否翻转 |
|---|---|---|---|
| -2 | 5.61 | -14.17 | 否 |
| -1 | 2.80 | -14.09 | 否 |
| 0 | 0 | -14.125 | — |
| +1 | 2.80 | -14.14 | 否 |
| +2 | 5.61 | -14.23 | 否 |

即使 alpha=±2（在 L18 残差流上注入约 7% 量级扰动），d_raw 仅移动 ~0.1。

### 2.2 最接近决策边界的 SS 错拒（dev group `a2c1a9...`）

该组 base d_raw=-0.75（全 dev 中最接近 0 的 SS 错拒）：

| alpha | d_raw |
|---|---|
| -2 | -0.6875 |
| -1 | -0.6875 |
| -0.5 | -0.6875 |
| +0.5 | -0.8125 |
| +2 | -0.8125 |

仍无法达到 d_raw>0（接受侧）。

### 2.3 极端干预也无法大幅移动（dev group `0a10ac...`）

| 干预 | d_raw |
|---|---|
| base | -3.0 |
| R_end += 100·noise (|Δ|=280) | -2.19 |
| R_end := 0 | -2.06 |
| R_end := 100 | -2.06 |

将单个 R_end 残差向量完全覆盖，也只移动 d_raw 约 0.9，远不足以翻转
SS 错拒（其 d_raw 分布中位数 ≈ -15.0，范围 -0.75 ~ -19.78）。

## 3. 方向可预测性与弱因果性并存（不矛盾）

- train 上 `z = v*·(h - mu_train)` 的类间差距：y1 均值 - y0 均值 ≈ 6.15，
  即约 2.19 个 sigma（AUROC 0.922）。
- 但该方向位于 L18/R_end 的**状态空间**，其与最终 logits 的传递路径经过
  后续 ~10 层 residual 与归一化，单点加性扰动被大幅稀释。
- 结论：D2-R1 中 L18/R_end 的风险信号是**指示性（predictive）**的，
  在本设定的"单体前向、L18×R_end、线性单方向、加性 patch"方法下，
  不具备足够的选择性因果作用。这与 D3 的结论边界一致：只否定该具体方法，
  不宣称参数知识非唯一原因，也不宣称已发现普适机制方向。

## 4. 受影响交付文件

- `dev_risk_selection_and_grid.csv`：32 个配置全部 SS_net_gain=0、harm=0、CSI=0。
- `dev_configuration_freeze.json`：frozen=false。
- Phase 3 各文件：NOT RUN 占位，final-reserve 全程未被读取。
