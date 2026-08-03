# inheritance_audit.md

## E0 / E0-R1 / E0-R2 → E1 继承对账

| 项 | 值 | 状态 |
|---|---|---|
| E0 label | popqa_relation_swap_capacity_insufficient | ✓ |
| E0-R1 label | popqa_relation_coverage_insufficient | ✓ |
| E0-R2 label | popqa_relation_swap_external_data_qualified | ✓ |
| E0-R2 h1_development_approved | True | ✓ |
| total == 14077 | 14077 | ✓ |
| split counts 8446/2815/2816 | {'train': 8446, 'dev': 2815, 'final_reserve': 2816} | ✓ |
| fixed index covers 14077 | 14077 | ✓ |
| dev rows == 2815 | 2815 | ✓ |
| non-dev counts match (train 8446 / final_reserve 2816) | {'train': 8446, 'final_reserve': 2816} | ✓ |
| no unknown split rows | 0 | ✓ |
| dev group ids unique | 2815 | ✓ |
| dev ids all in fixed index with split=dev |  | ✓ |
| donors within dev split | 0 violations | ✓ |
| r_o != r_s on dev (normalized) | 0 violations | ✓ |
| T0 render contract on dev | 0 violations | ✓ |
| dev covers all 16 relations | 16 | ✓ |
| dev relations exact official set |  | ✓ |
| dev group_count matches E0-R2 approved | 2815 vs 2815 | ✓ |
| dev sorted_group_id_sha256 matches E0-R2 approved | 14aa6be52f0698aa… vs 14aa6be52f0698aa… | ✓ |
| dev relation_distribution_sha256 matches E0-R2 approved | d94d48037e62e825… vs d94d48037e62e825… | ✓ |
| dev text input written (2815) | 2815 dev-only rows | ✓ |
| prompt constants present |  | ✓ |
| accept/reject ids 362/425 |  | ✓ |
| revision a09a3545... | a09a35458c702b33eeacc393d103063234e8bc28 | ✓ |
| hash config.json | 7463bb0ea7831536… vs 7463bb0ea7831536 | ✓ |
| hash tokenizer.json | c0382117ea329cdf… vs c0382117ea329cdf | ✓ |
| hash tokenizer_config.json | 5b5d4f65d0acd3b2… vs 5b5d4f65d0acd3b2 | ✓ |
| hash vocab.json | ca10d7e9fb3ed185… vs ca10d7e9fb3ed185 | ✓ |
| hash merges.txt | 599bab5407508877… vs 599bab5407508877 | ✓ |
| hash model.safetensors.index.json | 624bf7c47cd12468… vs 624bf7c47cd12468 | ✓ |

## dev relation 分布（描述性）

| relation | dev group 数 |
|---|---|
| author | 292 |
| capital | 120 |
| capital of | 70 |
| color | 4 |
| composer | 201 |
| country | 169 |
| director | 419 |
| father | 106 |
| genre | 324 |
| mother | 39 |
| occupation | 112 |
| place of birth | 114 |
| producer | 294 |
| religion | 70 |
| screenwriter | 389 |
| sport | 92 |

## 边界

- 本轮只评分 PopQA dev（2,815 group）；未读取/评分任何 train/final-reserve。
- 未读取 hidden state；未训练 Probe；无干预。
