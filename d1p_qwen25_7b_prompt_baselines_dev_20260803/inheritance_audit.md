# inheritance_audit.md

## Phase 0 继承与模板合同

| 项 | 状态 |
|---|---|
| D0 label | ✓ |
| D1 label | ✓ |
| D1-R label | ✓ |
| D1-R-A label | ✓ |
| T0 canonical + SHA256 | ✓ |
| T1 canonical + SHA256 | ✓ |
| T2 canonical + SHA256 | ✓ |
| model hash config.json | ✓ |
| model hash tokenizer.json | ✓ |
| model hash tokenizer_config.json | ✓ |
| model hash vocab.json | ✓ |
| model hash merges.txt | ✓ |
| model hash model.safetensors.index.json | ✓ |
| model revision | ✓ |
| accept/reject ids | ✓ |
| D1 system prompt (verbatim) | ✓ |
| D1 user template (verbatim) | ✓ |
| D1 accept/reject continuations | ✓ |

## B_direct / B_CoT_gen 指令（预注册，不可调整）

- `B_direct`：在 system 既有任务说明之后、Question 字段之前（即 system message 尾部）追加固定指令（见 `baseline_prompt_spec.json`）。
- `B_CoT_gen`：以 B_direct 为基础，再在任务说明最后追加固定 CoT 指令；greedy 生成 `max_new_tokens=128`、无 stop 后处理；解析仅接受最后一个非空行严格等于 `Final verdict: A` / `Final verdict: B`。
- 两者均不修改 Candidate 模板、Reference swap、A/B 标签定义或题组切分。

## 只读范围

- dev-only 数据：D1 `scripts/_dev_pairs.jsonl`（195 groups）。
- 未读取 train / final-reserve 文本；未提取 hidden states；未训练 Probe。
