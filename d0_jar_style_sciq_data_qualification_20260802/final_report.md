# Final Report — E01-D0 JAR-style reference–knowledge conflict in LLM Judges：SciQ 数据资格门

| 问题 | 结果 |
|---|---|
| SciQ 官方数据是否可获取且可哈希固化？ | 是（hf-mirror 官方镜像；revision `2c94ad3e…`；validation parquet SHA256 已固化） |
| validation schema 是否满足数据契约？ | 是（1000 行 = 1000 唯一 source group；question/correct_answer/distractor1-3/support 全部字段无缺失） |
| 是否构造出完整四格 pair？ | 是（979 组，每组四格逻辑实例齐备，`r_o≠r_s` 且渲染后 `c_o≠c_s`） |
| 机械过滤与 swap 规则是否可复现？ | 是（固定 seed 20260802/20260803；两次运行全部产物 SHA256 一致） |
| 保留的唯一 source group 数量 | 979（≥ 600） |
| train/dev/final-reserve 是否 group 隔离？ | 是（587 / 195 / 197；979/979 组各属唯一 split，无泄漏） |
| 是否加载任何模型或运行 Judge？ | 否（`model_loaded=false`，`judge_scored=false`，`hidden_states_read=false`，`probe_trained=false`） |
| 是否允许进入 D1 行为资格门？ | 是（数据资格可行；D1 为独立后续门，本轮不执行） |
| 最终标签 | `jar_style_sciq_data_qualification_feasible` |

## 一、数据来源与访问审计

- 数据集：`allenai/sciq`（HF datasets repo，官方公开数据），官方主页 https://allenai.org/data/sciq。
- 本地缓存无 SciQ；经官方镜像 `hf-mirror.com`（`hf_hub_download`，`repo_type='dataset'`）下载，仅数据文件本身。
- revision (commit)：`2c94ad3e1aafab77146f384e23536f97a4849815`。
- 下载时间：2026-08-02（本地时区 UTC+8）。
- 文件 SHA256（validation）：
  `455dd9f1d725cd3ecbce369799a2fbbdbbfecf51ab84a86d56ba3370dc847b8a`
- 许可：CC BY-NC 3.0（学术研究用途）。
- 详见 `source_access_audit.md`。

## 二、数据契约

- 字段：`question`、`correct_answer`、`distractor1`、`distractor2`、`distractor3`、`support`（887/1000 非空）。
- source group = 一个原始 question；`source_group_id = SHA256(NFKC(question)‖NFKC(correct_answer)‖NFKC(d1)‖NFKC(d2)‖NFKC(d3))`，分隔符 `|||`，不使用行号。
- validation：1000 行 → 1000 个唯一 source_group_id。
- 归一化：NFKC → trim → 连续空白折叠为单个空格（不做大小写转换）。
- 详见 `source_data_contract.md`。

## 三、候选渲染规则

```text
Candidate(answer) = "The answer is <answer>."
```

- 模板、大小写、标点固定；唯一允许变化的是 `<answer>` 字段（取 `r_o` / `r_s` 的归一化原文，不做大小写转换，保证 `r_o≠r_s ⇒ c_o≠c_s`）。
- 已写入 `candidate_rendering_spec.json`（含 UTF-8 原文 SHA256：`a5d8d816…`）。

## 四、机械过滤漏斗（pre-registered，不做人工/LLM/embedding/外部知识挑选）

| 规则 | 排除数 | 累计 |
|---|---:|---:|
| r1 r_o 非空 | 0 | 0 |
| r2 r_o 为 1–6 个空白分隔 token | 1 | 1 |
| r3 r_o 不含数字 | 7 | 8 |
| r4 不含换行/URL/选项编号/括号型多选标记 | 3 | 11 |
| r5 只含英文字母/空白/连字符/撇号 | 3 | 14 |
| r6 非泛化答案（yes/no/true/false/unknown/none） | 0 | 14 |
| r7 r_o 不以完全规范化形式出现在 question | 7 | 21 |
| r8 r_o 不与任一 distractor 相同 | 0 | 21 |
| **pass_all（进入 swap 候选池）** | **979** | 21 |

- 被排除 21 组，全部为机械规则排除；未因任何难度/语义印象重抽。

## 五、Coarse-form-controlled Random Swap

- 候选 `r_s` 仅从另一保留组的 `r_o` 中选出，约束：`r_s≠r_o`、不等于本题任何 distractor、不以完全规范化形式出现在本题 question 或 support、不与本题答案字段重复、pair 内组不同、每组仅保留一个 `r_s`。
- 选择算法：`random.Random(20260802).choice(排序后候选列表)`，组按 `source_group_id` 升序处理；每次重跑结果一致。
- 979 组全部存在可行 `r_s`（swap 排除 0 组），即 coarse 形态约束未耗尽 SciQ validation 容量。
- 不称其为同实体类型 swap / Plausible Swap / JAR 原始数据复现。

## 六、四格样本定义（协议标签，非世界事实）

| reference | candidate | expected label |
|---|---|---|
| `r_o` | `c_o` | `Correct` |
| `r_o` | `c_s` | `Incorrect` |
| `r_s` | `c_o` | `Incorrect` |
| `r_s` | `c_s` | `Correct` |

- 979/979 组四格齐备；`r_o≠r_s`、`c_o≠c_s` 全量验证通过。
- 本轮不宣称替换参考在世界知识上错误、不宣称与任何 Judge 参数知识冲突、不宣称候选自然语言语义经人工验证。

## 七、容量与切分

- 保留唯一 group：**979**（≥ 600 容量门，通过）。
- 以 `source_group_id` 为最小切分单位，固定 seed `20260802` shuffle 后按序切分：

| split | groups | 比例 |
|---|---:|---:|
| train | 587 | 60.0% |
| dev | 195 | 19.9% |
| final_reserve | 197 | 20.1% |

- 组级隔离验证：979/979 组各属唯一 split；`fixed_split_indices.json` 记录 seed、排序方法、各 split 的组 id 与 SHA256。
- `final_reserve` 仅写入 sealed manifest（`fixed_split_indices.json`），本轮不做模型评测。

## 八、盲语义审计包

- 从最终 pool 以固定 seed `20260803` 抽样 100 组，字段仅 `question, r_o, r_s, c_o, c_s`。
- 不含 split、模型名、分数、行为标签或任何历史实验信息。

## 九、可复现性

- 流水线脚本 `scripts/d0_pipeline.py`（固定 seeds 20260802 / 20260803）。
- 两次独立运行产物（`preliminary_swap_pairs.jsonl`、`fixed_split_indices.json`、`filter_funnel.csv`、`blind_semantic_audit_packet.csv`、`candidate_rendering_spec.json`）SHA256 完全一致。

## 十、结论边界

- 本轮仅证明 SciQ 四格构造的机械可行性、文本条件与容量可审计；
- 未加载任何模型、未运行 Judge、未读取 hidden state、未训练任何方法；
- 不得据本结果直接声称已具备 reference–knowledge 冲突现象或机制证据。

**最终标签：`jar_style_sciq_data_qualification_feasible`**
