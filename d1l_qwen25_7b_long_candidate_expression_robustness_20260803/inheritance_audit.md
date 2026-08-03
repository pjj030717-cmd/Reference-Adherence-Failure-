# inheritance_audit.md

## Phase 0 继承对账

| 项 | 值 | 状态 |
|---|---|---|
| D0 label | jar_style_sciq_data_qualification_feasible | ✓ |
| D1 label | jar_style_reference_override_behavior_feasible | ✓ |
| D1-R label | template_robust_reference_override_feasible | ✓ |
| D1-R-A label | template_provenance_and_diversity_audit_complete | ✓ |
| canonical T0 | The answer is <answer>. | ✓ |
| canonical T1 | For this question, the answer is <answer>. | ✓ |
| canonical T2 | The response is <answer>. | ✓ |
| tokenize ' A' == [362] single token | [362] | ✓ |
| decode ' A' roundtrip | ' A' | ✓ |
| ' A' no UNK |  | ✓ |
| tokenize ' B' == [425] single token | [425] | ✓ |
| decode ' B' roundtrip | ' B' | ✓ |
| ' B' no UNK |  | ✓ |
| revision == D1 a09a3545... | a09a35458c702b33eeacc393d103063234e8bc28 | ✓ |
| hash config.json | 7463bb0ea7831536… vs recorded 7463bb0ea7831536 | ✓ |
| hash tokenizer.json | c0382117ea329cdf… vs recorded c0382117ea329cdf | ✓ |
| hash tokenizer_config.json | 5b5d4f65d0acd3b2… vs recorded 5b5d4f65d0acd3b2 | ✓ |
| hash vocab.json | ca10d7e9fb3ed185… vs recorded ca10d7e9fb3ed185 | ✓ |
| hash merges.txt | 599bab5407508877… vs recorded 599bab5407508877 | ✓ |
| hash model.safetensors.index.json | 624bf7c47cd12468… vs recorded 624bf7c47cd12468 | ✓ |
| prompt constants loaded |  | ✓ |
| accept/reject ids |  | ✓ |
| dev pairs == 195 | 195 | ✓ |
| no final-reserve/train text opened | streamed dev-only from _dev_pairs.jsonl | ✓ |
| D1 four_cell_scores_dev.csv rows == 780 | 780 | ✓ |
| D1 four-cell cells correct |  | ✓ |
| D1 dev groups == 195 |  | ✓ |
| T0 template fixed sha | c42e1ea10a6be334… | ✓ |
| T0 single placeholder + no forbidden words | banned=[] | ✓ |
| T3 template fixed sha | b9d4ba1fcb70a626… | ✓ |
| T3 single placeholder + no forbidden words | banned=[] | ✓ |
| T4 template fixed sha | 068fdfd1871f32bf… | ✓ |
| T4 single placeholder + no forbidden words | banned=[] | ✓ |
| T5 template fixed sha | ee24f106d0b0a76f… | ✓ |
| T5 single placeholder + no forbidden words | banned=[] | ✓ |
| candidate contract: no violations | 0 violations | ✓ |
| contract rows == 195*4 templates | 780 | ✓ |

## 模板固定声明

- T0（复现基准）SHA256 `c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc`
- T3-bare SHA256 `b9d4ba1fcb70a6267f1016f0b8a13e739a6f1a6c4145b0526b98a262f2a2e4f1`
- T4-long-first SHA256 `068fdfd1871f32bffb7ca9705222f71410b90abfcd41c99b1b2cb920c5e283b5`
- T5-long-last SHA256 `ee24f106d0b0a76f79709f99782fd35ee7b1267f1565b8bb37e8064747e21020`
- 以上在**任何行为结果之前**写入 `candidate_length_expression_spec.json`；不得事后修改。

## 禁止项遵守

- 未读取 D0 train/final-reserve 样本文本（仅读 D0 decision.json 标签）。
- 未读取/评分/缓存任何 final-reserve。
- 未读取 hidden state；未训练 Probe/Monitor/Classifier；无干预/hook/causal tracing。
- 未修改 system prompt / prompt baseline / CoT / ICL / SFT；未换模型/精度/batch/读出。
- 未用 LLM 改写或审查 Candidate；未拼接 SciQ `support`；未按行为结果选择样本。
