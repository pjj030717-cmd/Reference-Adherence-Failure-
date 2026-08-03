# protocol_amendment_e0_to_e0r1.md

## E0 原样保持

- E0 最终标签 `popqa_relation_swap_capacity_insufficient` 及其有效停止结论保持不变，不得修改。
- E0 已生成的所有数据、过滤、split、donor、四格、模板工件不因本修正而改变。

## 修改前后 relation 规则

| | E0 | E0-R1 |
|---|---|---|
| 每 split relation/property 数 | ≥ 20 | 覆盖全部 16 个官方 relation/property |
| 每 relation 每 split 保留组 | 未单独要求 | ≥ 10 |
| 每 split 最大 relation 占比 | ≤ 0.25 | ≤ 0.25 |

## 修改理由

- E0 经官方 schema 确认 PopQA 数据仅包含 16 个 relation/property 类型（director, screenwriter, genre,
  producer, author, composer, country, capital, place of birth, father, sport, occupation, capital of,
  religion, mother, color）。
- 因此 E0 的“每 split ≥ 20 类 relation”门槛在数学上不可满足（数据集内不存在 20 类），属于数据集—门槛不匹配。
- E0-R1 将门槛改为“覆盖该数据集完整的官方 relation universe（16 类）+ 每类每 split ≥ 10 组 + 最大占比 ≤ 0.25”。

## 不改变的内容

- 数据源：akariasai/PopQA，revision 098765c7，test.tsv SHA256 9a5227f4…
- 过滤规则 R1–R8、split seed 20260816、60/20/20、split 内同 relation donor、每 group 独立 RNG `20260816|sgid`、
  四格构造、T0/T1/T2 模板（均逐字继承，不做任何修改）。

## 范围声明

- E0-R1 只验证外部数据资格；不含任何 Judge 加载、模型前向、打分、hidden state、Probe 或行为结论。
