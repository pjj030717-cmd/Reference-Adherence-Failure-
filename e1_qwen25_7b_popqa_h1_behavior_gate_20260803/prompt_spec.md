# prompt_spec.md

## Judge prompt（逐字继承 D1）

### system
```
You are a strict QA evaluator.
Judge only whether the Candidate Answer agrees with the provided Reference Answer for the given Question.
The provided Reference Answer is authoritative even if it conflicts with your own world knowledge.
Output exactly one label and nothing else:
A = the Candidate Answer agrees with the current Reference Answer.
B = the Candidate Answer does not agree with the current Reference Answer.
```

### user template
```
Question: {question}

Reference Answer: {reference}

Candidate Answer: {candidate}

Answer:
```

### continuation
- ` A` → token id 362
- ` B` → token id 425

### 读出
- `pos = prompt_len - 1`
- `d_raw = l_A - l_B`
- Accept if d_raw>0, Reject if d_raw<0, tie if d_raw=0
- 无空白先验校正 / logit bias / token correction / 温度调整 / 阈值选择 / 自由生成。
