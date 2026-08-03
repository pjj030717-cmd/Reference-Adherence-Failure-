# E0：PopQA Relation-Controlled Swap 外部数据资格门 — 最终报告

## 结果总表

| 问题 | 结果 |
|---|---|
| PopQA 官方数据是否可唯一获取？ | 是（hf-mirror，commit 098765c7，SHA256 已记录） |
| question / answer / relation-property 契约是否有效？ | 是（question/obj/prop/id，14,267 行唯一） |
| 是否在 split 内完成 relation-controlled donor 选择？ | 是（train/dev/final-reserve 各自 split 内，同 relation，r_s≠r_o） |
| 四格与 T0/T1/T2 Candidate 合同是否全部通过？ | 是（四格共享 q/r_o/r_s/c_o/c_s；模板逐字继承且 SHA256 一致） |
| 是否达到外部确认容量门槛？ | 否（total=14077，train=8446，dev=2815，final=2816；每 split relation={'train': 16, 'dev': 16, 'final_reserve': 16}；max share=0.1420） |
| 是否加载任何 Judge 或运行任何推理？ | 否 |
| 是否读取旧实验 train/final-reserve 文本？ | 否 |
| 最终标签 | popqa_relation_swap_capacity_insufficient |

## 结果

容量门未通过：relations_per_split {'train': 16, 'dev': 16, 'final_reserve': 16} (PopQA 官方仅 16 个 relation 类型)

## 方法

- 数据：PopQA 官方 `test` split（14,267 行），字段 question / obj / prop / id。
- source_group_id = SHA256(NFKC(q) ∥ '\x00' ∥ NFKC(obj) ∥ '\x00' ∥ NFKC(prop) ∥ '\x00' ∥ NFKC(str(id)))，唯一。
- 机械过滤 R1–R6（记录级）→ R7/R8（split 内 donor 相关）；R4 排除 1 组，R6 排除 189 组。
- 先切分（dict 排序 + Random(20260816) shuffle，60/20/20），再在 split 内选 donor（同 relation、sgid 不同、答案规范化后不同；每 group 独立 RNG `20260816|sgid`，choice 一次）。
- 四格合同：OO/OS/SO/SS 各用 r_o/r_s 与 c_o/c_s，全部机械验证。
- Candidate 模板仅继承 D1-R-A canonical T0/T1/T2，逐字与 SHA256 对照，无新模板。

## 过滤漏斗

| 阶段 | 排除 | 剩余 |
|---|---|---|
| 初始 | 0 | 14,267 |
| R1 空字段 | 0 | 14,267 |
| R2 控制字符 | 0 | 14,267 |
| R3 答案长度 | 0 | 14,267 |
| R4 答案 token 数 | 1 | 14,266 |
| R5 问题 token 数 | 0 | 14,266 |
| R6 答案出现在问题中 | 189 | 14,077 |
| R7 split 内 relation<2 | 0 | 14,077 |
| R8 donor 答案相同 | 0 | 14,077 |
| 最终保留 |  | 14,077 |

## 容量

- 总保留 14,077 ≥ 1,200 ✓；train 8,446 ≥ 720 ✓；dev 2,815 ≥ 240 ✓；final-reserve 2,816 ≥ 240 ✓。
- **每 split distinct relation = 16 < 20 ✗**（PopQA 官方仅 16 个 relation/property 类型，任何 split 无法达到 ≥20）。
- 最大 relation 占比 0.1420 ≤ 0.25 ✓。

按协议，容量不足时不得放宽过滤、改变切分比例、换 seed、跨 split 选 donor、引入第二数据集或降低门槛；
因此输出 `popqa_relation_swap_capacity_insufficient` 并停止。

## 盲审计包

容量门未通过，不生成 blind_candidate_contract_packet.csv。

## 边界与局限

- PopQA 官方仅单一 split；本实验按协议 2.1 自行切分 train/dev/final-reserve。
- 本构造是 JAR-style、relation-controlled 的外部 swap，不声称复现 JAR 原始 type-preserving pipeline。
- license 未声明（not specified）。
