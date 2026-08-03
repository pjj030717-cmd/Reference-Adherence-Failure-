# Final Report — E1 Qwen2.5-7B × PopQA H1 Reference-Adherence Failure 行为资格门

## 结果总表

| 问题 | 结果 |
|---|---|
| E0 / E0-R1 / E0-R2 是否可唯一继承？ | 是（三标签 + 14077/8446/2815/2816 + dev manifest 与 E0-R2 批准值一致） |
| 是否只读取并评分 PopQA dev？ | 是（2815 group × 4 cell = 11,260 判断） |
| train / final-reserve 是否未暴露给模型和结果工件？ | 是（静默 split filter；未 tokenize/前向/评分/缓存） |
| 模型与 A/B teacher-forced 读出是否有效？ | 是（revision a09a3545…，hash 一致，pos=prompt_len-1） |
| 24 条 synthetic regression 是否通过？ | 是（24/24；A 12/12；B 12/12；ties=0；greedy 一致 24/24） |
| OO / OS / SO / SS 的结果 | 0.996 / 1.000 / 0.997 / 0.947 |
| SS false rejection 是否达到资格门？ | 否（FR_SS=0.051，CI [0.043, 0.060]；SS group=144） |
| 是否允许进入 PopQA H2 协议设计？ | 否 |
| 最终标签 | popqa_h1_behavior_insufficient |

## 1. 继承与数据

- E0（`popqa_relation_swap_capacity_insufficient`）、E0-R1（`popqa_relation_coverage_insufficient`）、
  E0-R2（`popqa_relation_swap_external_data_qualified`）原结论原样保留并唯一核验。
- 总保留 14,077；train/dev/final-reserve = 8,446/2,815/2,816。
- dev manifest 与 E0-R2 approved manifest 完全一致：
  `sorted_group_id_sha256=14aa6be5…`，`relation_distribution_sha256=d94d4803…`。
- dev 覆盖 16/16 relation；donor 同 split、同 relation、不同 group；r_o != r_s；T0 渲染合同有效。

## 2. dev-only 隔离

```text
source_stream_scanned_for_split_filter = true
final_reserve_text_exposed_to_model = false
final_reserve_model_scored = false
final_reserve_hidden_state_read = false
train_text_exposed_to_model = false
```

- 从 `E0-R1 external_swap_pairs.jsonl` 静默、机械按 `split` 字段过滤；非 dev 行计数后立即丢弃，
  未打印、未保存、未抽样、未统计其文本；未对 non-dev 文本做 tokenization/前向/评分。

## 3. 模型与读出回归

- 模型：Qwen2.5-7B-Instruct，revision `a09a3545…`，config/tokenizer/index hash 与 D1 逐位一致；BF16/eval/inference_mode/batch=1。
- ` A`→id 362、` B`→id 425（单 token、无 UNK）；`pos = prompt_len - 1`。
- 24 条合成对：24/24，A 12/12，B 12/12，ties=0，greedy 一致 24/24（`synthetic_readout_audit.csv`）。

## 4. PopQA dev 四格行为（2815 group，micro-average）

| cell | n | accuracy | accept_rate | mean d_raw | median d_raw | p05 | p25 | p75 | p95 | tie |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OO | 2815 | 0.9964 | 0.9964 | 16.4299 | 16.6309 | 14.5283 | 15.9375 | 17.3242 | 18.3750 | 0.0000 |
| OS | 2815 | 0.9996 | 0.0004 | -20.8945 | -21.0312 | -22.6875 | -21.7500 | -20.2500 | -18.8125 | 0.0000 |
| SO | 2815 | 0.9968 | 0.0032 | -19.8029 | -20.4375 | -22.2188 | -21.2812 | -19.1875 | -15.4644 | 0.0000 |
| SS | 2815 | 0.9474 | 0.9474 | 14.6811 | 16.4453 | -0.6250 | 15.4922 | 17.2734 | 18.4844 | 0.0014 |


```text
ACC_o  = 0.9980
ACC_s  = 0.9721
RPAG   = 0.0259
FR_SS  = 0.0512   (SS false-reject group 数 = 144 / 2815)
FA_SO  = 0.0032
总 tie rate = 0.0004
d_raw 全局 mean=-2.3966 median=-11.9062
```

### Bootstrap（2,000 次 source-group 重采样，seed=20260819，95% CI）

| metric | 95% CI |
|---|---|
| FR_SS | [0.0433, 0.0597] |
| RPAG | [0.0218, 0.0306] |

## 5. relation/property 描述性审计（不用于资格门）

- 全部 16 类报告 group 数（见 `relation_descriptive_audit.csv`）。
- 仅 `n >= 30` 的 relation 报告 FR_SS 与 95% CI。
- `color`（dev 仅 4 组）只报告样本数，不报告 CI、不作任何 relation 结论。
- 例如：capital 120 组 FR_SS=0.625（CI [0.533, 0.717]）；author 292 组 FR_SS=0.007（CI [0.000, 0.017]）；
  screenwriter 389 组 FR_SS=0.005。relation 差异仅作描述，不进入任何门或筛选。

## 6. H1 行为资格门判定

| 门 | 值 | 通过 |
|---|---|---|
| OO accuracy ≥ 0.85 | 0.9964 | ✓ |
| OS accuracy ≥ 0.85 | 0.9996 | ✓ |
| ACC_o ≥ 0.85 | 0.9980 | ✓ |
| FR_SS ≥ 0.25 | 0.0512 | ✗ |
| bootstrap CI lower(FR_SS) ≥ 0.20 | 0.0433 | ✗ |
| RPAG ≥ 0.15 | 0.0259 | ✗ |
| SS false-reject group ≥ 200 | 144 | ✗ |
| tie rate ≤ 0.02 | 0.0004 | ✓ |
| 无 NaN/inf（11260 行） | True | ✓ |

**最终标签：`popqa_h1_behavior_insufficient`**

## 7. 解释边界

- FR_SS = 0.0512 表示：PopQA dev 中 Candidate 与 swapped Reference 一致、但 Judge 仍错误 Reject 的比例 ≈ 5.1%（144/2815）。
- 该量级不足以满足 H1 资格门（需要 ≥ 0.25 且 CI lower ≥ 0.20）：与 SciQ 上观测到的高 SS 错拒（D1 中 FR_SS≈0.76）形成对照。
- 不证明错误由参数知识导致；不证明该现象跨模型普适；不授权 hidden-state/Probe。
- 未读取 final-reserve、未读取 hidden state、未训练 Probe、无任何干预。
- 本轮停止；不得自动进入 H2。
