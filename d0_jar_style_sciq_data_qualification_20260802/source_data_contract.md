# source_data_contract.md

## 字段（SciQ validation split，parquet schema）
| 字段 | 类型 | 说明 |
|---|---|---|
| question | string | 原始问题 |
| correct_answer | string | 唯一正确答案（r_o 来源） |
| distractor1/2/3 | string | 三个干扰项 |
| support | string | 支持性证据段落（887/1000 非空；113 行为空字符串） |

## source group 定义
- 每个原始 question 为一个 source group（validation split 共 1000 个）。
- source_group_id = SHA256(NFKC(question) || "|||" || NFKC(correct_answer) || "|||" || NFKC(distractor1) || "|||" || NFKC(distractor2) || "|||" || NFKC(distractor3))。
- 分隔符使用 "|||" 以避免字段拼接歧义；不使用行号。
- 1000 行 → 1000 个唯一 source_group_id（无重复）。

## 归一化规则（机械过滤与哈希共用）
- NFKC → trim → 连续空白折叠为单个空格。
- 不做大小写转换（保证 r_o != r_s 时渲染后 c_o != c_s）。

## revision 与哈希
- revision：`2c94ad3e1aafab77146f384e23536f97a4849815`
- validation parquet SHA256：`455dd9f1d725cd3ecbce369799a2fbbdbbfecf51ab84a86d56ba3370dc847b8a`

## 任何不确定性
- support 缺失 113 行：机械过滤仅在 r_s 约束中使用 support 子串检查，空 support 视为无约束。
- 许可证为 CC BY-NC 3.0（非商业），本实验为学术研究用途。
