# tokenization_audit.md

## Continuation Tokenization 审计

| 项 | accept " A" | reject " B" |
|---|---|---|
| tokenize() | ['ĠA'] | ['ĠB'] |
| encode() | [362] | [425] |
| decode | ' A' | ' B' |
| 单 token | 是 | 是 |
| continuation length | 1 | 1 |
| UNK | 否 | 否 |

- 两者 continuation length 相同：True（均为 1）
- token id 不同：True
- 无 UNK / 空序列 / 额外 special token：True

**结论：decision channel 公平可评分。**
