# D1-R-A：候选模板溯源与差异强度审计

## 问题与结果

| 问题 | 结果 |
|---|---|
| T0/T1/T2 是否可唯一恢复？ | 是（3 来源逐字一致，SHA256 唯一） |
| T0 是否与 D1/D0 基础渲染一致？ | 是（D0 rendering spec / D1 dev pairs / D1-R T0 复现 780 行全部一致） |
| 是否存在显式 Reference/评分引导污染？ | 否（reference/correct/valid 引导词全部为 0） |
| 模板差异属于何种强度？ | **structural_candidate_variation** |
| 是否运行了任何 Judge 推理？ | 否（仅加载 tokenizer） |
| 是否读取了 train/final-reserve 文本？ | 否 |
| 最终标签 | **template_provenance_and_diversity_audit_complete** |

## 模板（canonical，来自 D1-R 可执行来源）

| 模板 | 原始 UTF-8 | SHA256 | 占位符 |
|---|---|---|---|
| T0 | `The answer is <answer>.` | `c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc` | 1×`<answer>` |
| T1 | `For this question, the answer is <answer>.` | `d325f862ad174533fe38c193744bbebd30b23e2ec72905a173c2b2eaed8fc078` | 1×`<answer>` |
| T2 | `The response is <answer>.` | `5fb1b5ed1ba1cb158981aea1673d936dcf88ff91b1423c6796031886de47df24` | 1×`<answer>` |

来源：D1-R `candidate_template_robustness_spec.json` + `scripts/d1r_template_spec.py` + `scripts/d1r_eval.py`（逐字一致）。

## 机械差异

| 对 | 字符 Levenshtein | 固定词 token Jaccard | 最长公共固定子串 | T 词数 | T 标点 |
|---|---|---|---|---|---|
| T0↔T1 | 20 | 0.375 | 22 | 4↔7 | `.` ↔ `,`+`.` |
| T0↔T2 | 7 | 0.600 | 13 | 4↔4 | `.` ↔ `.` |
| T1↔T2 | 21 | 0.222 | 13 | 7↔4 | `,`+`.` ↔ `.` |

- T2 与 T0 差异最小（换词 `answer`→`response`）；T1 与两者差异最大（新增 `For this question, ` 前缀）。

## 渲染审计

6 个 probe answer（`Paris`/`heart`/`H2O`/`800`/`true`/`Gianluigi Buffon`）渲染后记录字符数、词数、句子数、Qwen token ids 与 token 数（见 `template_rendering_audit.csv`、`tokenization_audit.md`）。

## 表达类型与预注册归类

| 模板 | answer_only | declarative | question_restatement | reference | eval/correct | multi_sentence |
|---|---|---|---|---|---|---|
| T0 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| T1 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| T2 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |

归类（仅基于 Phase 1 模板属性，未参考任何行为结果）：

- **`prompt_intervention_contamination`**：否。所有模板固定文本均无 reference/gold/correct/valid/should accept 等引导词。
- **`structural_candidate_variation`**：是。T1（`For this question, the answer is <answer>.`）把答案放入问题语境框架（question-restatement 结构），不再只是孤立答案陈述；T2 为换词表面变化；T0 为基准。
- **`minimal_surface_variation`**：T0↔T2 符合（仅换词），但 T1 的存在使整体归类升级为 `structural_candidate_variation`。

结论：**structural_candidate_variation**。T1/T2 可支持“跨 Candidate 表达稳健性”的较强证据（T1 改变答案呈现框架），且无 prompt intervention 污染；若仅比较 T0↔T2，则属于最小表面变化。

## 结论边界

- 本审计只回答“已用于 D1-R 的 T0/T1/T2 能否唯一恢复 + 差异强度”，不检验 H1/H2，不构成行为结论。
- 归类严格由模板固定文本属性决定；未使用 D1-R 准确率、SS 错拒率或任何后续结果。
- T1 的 `For this question` 是对问题语境的显式引用，但**不含** reference/correct/评分规则字样，故不构成 prompt intervention。
