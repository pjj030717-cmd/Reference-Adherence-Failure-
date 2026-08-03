# inheritance_audit.md

## 继承对账（Phase 0）

| 项 | 值 | 状态 |
|---|---|---|
| D0 final_label | `jar_style_sciq_data_qualification_feasible` | ✓ |
| D1 final_label | `jar_style_reference_override_behavior_feasible` | ✓ |
| D0 dev groups | 195（seed 20260802） | ✓ |
| D1 评分 group == D0 dev group | 一致 | ✓ |
| Qwen revision | `a09a3545…` | ✓ |
| config.json / tokenizer.json / vocab / merges / index hashes | 与 D1 model_access_audit.md 一致 | ✓ |
| system prompt / user 模板 / chat template / continuations `" A"`/`" B"` | 与 D1 `_prompt_constants.json` 一致 | ✓ |
| BF16 / eval / inference_mode / batch=1 / teacher-forced pos=prompt_len-1 | 继承 D1 实现 | ✓ |
| D1 synthetic readout audit | 24/24 | ✓ |
| 基础候选模板 T0 | `The answer is <answer>.`（SHA256 `c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc`） | ✓ |

**结论：D0/D1 唯一继承通过，可进入 T0 复现。**
