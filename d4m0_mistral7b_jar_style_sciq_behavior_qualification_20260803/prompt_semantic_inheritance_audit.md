# prompt_semantic_inheritance_audit.md

## 0.1 D0 继承审计摘要

| 审计项 | 值 | 通过 |
|---|---|---|
| D0 最终标签 | jar_style_sciq_data_qualification_feasible | ✓ |
| D0 dev group 数 | 195 | ✓ |
| D0 split seed | 20260802 | ✓ |
| D0 dev split SHA256 | 8be6f6f3450376cb90c597ad0fe9edf2ea422101501900c13c5604db61e4fd35 | ✓ |
| dev 195 组与 D0 split 索引一致 | 195 pairs, set-equal | ✓ |
| swap 映射唯一性（r_o != r_s，swap 源 != 本组） | 195/195 OK | ✓ |
| T0 Candidate 渲染语义（candidate 唯一由冻结 r_o/r_s + T0 模板构成） | 780-cell OK | ✓ |
| D0 冻结 c_o/c_s 字段复核（dev-only 文件） | 390 OK | ✓ |
| 四格期望标签映射 | OO=A, OS=B, SO=B, SS=A | ✓ |
| T0/T1/T2 模板 SHA256 复核 | T0/T1/T2 全部一致 | ✓ |

## 0.2 模板继承原则（唯一允许的序列化变化）

- 外层序列化仅允许一处变化：Qwen native chat template → Mistral 官方 native chat template
  （`tokenizer.apply_chat_template`）。其余（system/user 自然语言任务内容、字段顺序、
  verdict 定义、continuation ` A`/` B`、teacher-forced 位置）逐字继承 D1/D1-R。
- 不得把 Qwen 特殊 token / chat wrapper 硬塞给 Mistral；不得改写任务内容以"优化 Mistral"。

## D1 语义恢复（来自 D1 `scripts/_prompt_constants.json` 与 D1-R spec）

### system message 语义
```text
You are a strict QA evaluator.
Judge only whether the Candidate Answer agrees with the provided Reference Answer for the given Question.
The provided Reference Answer is authoritative even if it conflicts with your own world knowledge.
Output exactly one label and nothing else:
A = the Candidate Answer agrees with the current Reference Answer.
B = the Candidate Answer does not agree with the current Reference Answer.
```

### user message 字段顺序
```text
Question: {question}

Reference Answer: {reference}

Candidate Answer: {candidate}

Answer:
```

### verdict 定义
- A = Accept（Candidate 与 Reference 一致）
- B = Reject（Candidate 与 Reference 不一致）

### Candidate 表述（D1-R 冻结，逐字）
| 模板 | 字符串 | UTF-8 SHA256 |
|---|---|---|
| T0 | `The answer is <answer>.` | `c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc` |
| T1 | `For this question, the answer is <answer>.` | `d325f862ad174533fe38c193744bbebd30b23e2ec72905a173c2b2eaed8fc078` |
| T2 | `The response is <answer>.` | `5fb1b5ed1ba1cb158981aea1673d936dcf88ff91b1423c6796031886de47df24` |

D1-R spec 记录的模板哈希：
- T0 `c42e1ea1…`（本文件复核 `c42e1ea10a6be334…`）
- T1 `d325f862…`（本文件复核 `d325f862ad174533…`）
- T2 `5fb1b5ed…`（本文件复核 `5fb1b5ed1ba1cb15…`）

## Mistral native serialization 示例（tokenizer 渲染，不加载权重）

messages：
```python
[
  {"role": "system", "content": "…(D1 system 逐字)…"},
  {"role": "user", "content": "Question: …\n\nReference Answer: …\n\nCandidate Answer: …\n\nAnswer:"},
]
```

渲染结果（`add_generation_prompt=True`）：
```
<s>[INST] You are a strict QA evaluator.
Judge only whether the Candidate Answer agrees with the provided Reference Answer for the given Question.
The provided Reference Answer is authoritative even if it conflicts with your own world knowledge.
Output exactly one label and nothing else:
A = the Candidate Answer agrees with the current Reference Answer.
B = the Candidate Answer does not agree with the current Reference Answer.

Question: What is the capital of France?

Reference Answer: Paris

Candidate Answer: The answer is Paris.

Answer:[/INST]
```

首段 token ids（前 24 个）：[1, 3, 1763, 1228, 1032, 8881, 1954, 29509, 7316, 1796, 29491, 781, 29566, 25531, 1633, 3929, 1040, 24227, 27075, 28943, 1163, 1040, 4625, 23154]

## 继承结论

- 唯一继承源：D0（split/swap/模板语义/四格定义）→ D1（prompt/continuation/teacher-forced 语义）→ D1-R（T1/T2 表述）。
- 本轮目录不含任何 D0 train / final-reserve 完整题目文本；dev 文本仅经 D1 `_dev_pairs.jsonl`（dev-only）流式读取。
