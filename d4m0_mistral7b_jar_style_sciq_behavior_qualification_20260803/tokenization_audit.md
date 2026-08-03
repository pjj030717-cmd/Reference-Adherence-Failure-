# tokenization_audit.md

## 1.1 continuation tokenization（Mistral tokenizer，重新验证，不信旧值）

| 项 | 值 |
|---|---|
| `" A"` token ids | [1098] |
| `" B"` token ids | [1133] |
| 均为单 token | True / True |
| token id 不同 | True |
| 无 UNK | True |
| token 长度相等 | True |

accept_id = `1098`；reject_id = `1133`。

## 结论
- 通过：A/B continuation 是公平、语义正确的决策通道。
