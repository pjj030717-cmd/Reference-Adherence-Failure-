# template_provenance_audit.md

## 模板唯一来源恢复（Phase 0）

| 模板 | 原始 UTF-8 模板 | UTF-8 SHA256 | 来源数 | 逐字一致 |
|---|---|---|---|---|
| T0 | `The answer is <answer>.` | `c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc` | 3 | 是 |
| T1 | `For this question, the answer is <answer>.` | `d325f862ad174533fe38c193744bbebd30b23e2ec72905a173c2b2eaed8fc078` | 3 | 是 |
| T2 | `The response is <answer>.` | `5fb1b5ed1ba1cb158981aea1673d936dcf88ff91b1423c6796031886de47df24` | 3 | 是 |

来源（均逐字一致）：
- D1-R `candidate_template_robustness_spec.json` → `templates.<T>.template`
- D1-R `scripts/d1r_template_spec.py` → `TEMPLATES['<T>']`（可执行定义）
- D1-R `scripts/d1r_eval.py` → `TEMPLATES['<T>']`（可执行渲染路径）

## T0 与 D1/D0 基础渲染对齐

| 检查 | 结果 |
|---|---|
| D0 `candidate_rendering_spec.json` template == T0 | 是（`The answer is <answer>.`） |
| D1 `scripts/_dev_pairs.jsonl` c_o/c_s 全部 == T0 渲染 | 是（195 dev groups，0 违例） |
| D1-R `t0_reproduction_audit.csv` 780 行 candidate == T0 渲染 | 是（0 违例） |

注意：D0 `candidate_rendering_spec.json` 内嵌 `sha256_utf8` 字段为已知中间态哈希（`d41ad577…`），
D1-R `provenance_amendment.md` 已透明记录；模板**字符串**的 UTF-8 SHA256 为 `c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc`，以此继承。

## 占位符合同（Phase 0.2）

- 每模板恰有 1 个 `<answer>` 占位符，无其他占位符。
- `<answer>` 仅用于答案插入（D0 冻结 r_o/r_s 归一化原文）。
- 模板可确定性渲染。
- T0 ≠ T1 ≠ T2（逐字互异）。

## 机械差异（Phase 1.1，见 `template_pairwise_distance.csv`）

| 对 | 字符 Levenshtein | 固定词 token Jaccard | 最长公共固定子串 | 标点 |
|---|---|---|---|---|
| T0↔T1 | 20 | 0.375 | 22 | `.` vs `,`+`.` |
| T0↔T2 | 7 | 0.600 | 13 | `.` vs `.` |
| T1↔T2 | 21 | 0.222 | 13 | `,`+`.` vs `.` |

## 表达类型（Phase 1.3，见 `template_expression_type_audit.csv`）

| 模板 | answer_only | declarative_answer_frame | question_restatement | explicit_reference | eval/correct | multi_sentence | 污染提示 |
|---|---|---|---|---|---|---|---|
| T0 | 1 | 1 | 0 | 0 | 0 | 0 | 无 |
| T1 | 1 | 1 | 1 | 0 | 0 | 0 | 无 |
| T2 | 1 | 1 | 0 | 0 | 0 | 0 | 无 |
