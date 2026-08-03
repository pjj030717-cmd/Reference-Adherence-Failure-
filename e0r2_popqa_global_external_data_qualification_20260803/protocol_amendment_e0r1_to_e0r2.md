# protocol_amendment_e0r1_to_e0r2.md

## 修改内容

E0-R1 的规则“每个 relation/property 在每个 split 至少有 10 个保留 source group”在 E0-R2 中**不再作为资格门**。

## 理由（逐字记录）

1. PopQA 官方 relation universe 固定为 16 类，且关系分布存在天然长尾；
2. E0-R1 已确认所有 split 都覆盖完整 16 类，`color` 只是稀有类；
3. 本研究的外部确认单位是 source group，不是 relation；
4. relation 在该构造中承担“同 relation donor matching”的控制作用，而非分层独立统计检验；
5. 强制每个 split 每类至少 10 条，会把一个非主问题的 relation-level 统计要求错误提升为数据资格门；
6. 此修正不改任何数据、seed、split、donor、模板、四格或过滤规则。

## 不改变的内容

- 数据、过滤 R1-R8、split seed 20260816、60/20/20、donor（split 内同 relation）、四格、T0/T1/T2 全部不变。
- E0 与 E0-R1 的原停止结论原样保留。
