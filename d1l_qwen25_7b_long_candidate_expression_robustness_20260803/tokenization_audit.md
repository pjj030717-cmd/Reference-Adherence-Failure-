# tokenization_audit.md

## Continuation Tokenization 审计（与 D1 逐字一致）

| 项 | accept " A" | reject " B" |
|---|---|---|
| encode() | [362] | [425] |
| 单 token | 是 | 是 |
| UNK | 否 | 否 |

**结论：decision channel 公平可评分；与本轮模板渲染无关（A/B 为固定 continuation）。**
