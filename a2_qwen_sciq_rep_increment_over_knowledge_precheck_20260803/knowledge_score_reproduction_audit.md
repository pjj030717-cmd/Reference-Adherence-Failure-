# knowledge_score_reproduction_audit.md

## A1 冻结规范（完全继承，未改写）

- system：`Choose the option that is factually correct for the question.\nReply with only A or B.`
- user 模板：`Question:\n{q}\n\nOption A:\n{option_a}\n\nOption B:\n{option_b}\n\nAnswer:`
- 选项直接填 `r_o` / `r_s` 字符串，无 Candidate/Reference/Judge 句式。
- continuation：`" A"`=362、`" B"`=425（单 token、等长、非 UNK）。
- teacher-forcing 位置：`pos = prompt_len - 1`。
- Order1: A=r_o,B=r_s → d_1 = l_A − l_B；Order2: A=r_s,B=r_o → d_2 = l_B − l_A；k = (d_1+d_2)/2。
- 无空白先验校正、无阈值调参、无 prompt 搜索。

## 合成 24 对回归复现（A1 manifest 冻结原样）

| 检查 | 要求 | 本次结果 |
|---|---|---|
| overall | 24/24 | 24/24 |
| A-correct | 12/12 | 12/12 |
| B-correct | 12/12 | 12/12 |
| ties | 0 | 0 |
| greedy 一致 | >=22/24 | 24/24 |

## k 评分范围

- SciQ train：587 groups（`_k_train.json`）
- SciQ dev：195 groups（`_k_dev.json`）
- 无 NaN/inf；未接触 PopQA、final-reserve。
