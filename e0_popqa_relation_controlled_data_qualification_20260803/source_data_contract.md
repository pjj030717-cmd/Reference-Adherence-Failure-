# source_data_contract.md

## Schema 恢复（唯一确定）

| 概念 | 官方字段 | 官方含义（README） |
|---|---|---|
| question | `question` | PopQA question |
| 规范答案 | `obj` | object entity name（Wikidata object entity） |
| relation/property | `prop` | relationship type |
| 稳定官方记录 id | `id` | question id（唯一，14,267/14,267） |

`possible_answers` 为 gold answers 列表，用于评估；本轮以单一规范答案 `obj` 作为 canonical answer。

## source_group_id 定义

```
source_group_id = SHA256(
    NFKC(question)  || "\x00" ||
    NFKC(obj)       || "\x00" ||
    NFKC(prop)      || "\x00" ||
    NFKC(str(id))
)
```

- 分隔符：`\x00`（NUL，字符串中不可出现，防连接歧义）
- NFKC 直接采用（不做空白折叠），与协议 0.2 一致
- 唯一性：14,267 / 14,267 唯一
- 非空：question / canonical_answer / relation 均非空

## 规范化（用于过滤与比较）

```
norm(s) = ' '.join(NFKC(s).split())    # NFKC + 连续空白压缩为一个空格
保留大小写、保留标点
```

## 拟用 split

PopQA 官方仅提供单一 `test` split。本实验将其视为拟用全量池，随后按协议 2.1
以 source_group_id 排序 + `random.Random(20260816)` 打乱后切分 train/dev/final-reserve = 60/20/20。
