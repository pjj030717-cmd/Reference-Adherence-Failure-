# inheritance_audit.md

## 继承对账（Phase 0）

| 项 | 值 | 状态 |
|---|---|---|
| D0 final_label | `jar_style_sciq_data_qualification_feasible` | ✓ |
| D0 source revision | `2c94ad3e1aafab77146f384e23536f97a4849815` | ✓ |
| candidate template 字符串 | `The answer is <answer>.` | ✓ |
| template 字符串 SHA256（权威） | `c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc` | ✓ |
| candidate_rendering_spec.json 记录 sha256_utf8 | `d41ad577206de5b7c65465259bb2c89cf557062a7fd356aa23457b3c6ef0e06b`（见下方缺陷说明） | ⚠ 记录缺陷 |
| candidate_rendering_spec.json 最终全文 SHA256 | `a5d8d81611249cb076055a6a560b997996768a8f42fcea20041d607f8877f817` | ✓ |
| 979 个 pair 渲染均符合模板 | 是 | ✓ |
| split seed | 20260802 | ✓ |
| group 数（pairs） | 979 唯一 | ✓ |
| train / dev / final_reserve | 587 / 195 / 197 | ✓ |
| split 互斥且并集完整 | 是 | ✓ |
| 各 split SHA256 重算一致 | train 167d547f08659a57… / dev 8be6f6f3450376cb… / reserve 9fe440d6cb383c5c… | ✓ |
| dev 行流式过滤 | 195 行；train 丢弃 587；reserve 丢弃 197 | ✓ |

## D0 缺陷披露（不影响本门）

- D0 的 `candidate_rendering_spec.json` 中 `sha256_utf8` 字段（`d41ad577206de5b7c65465259bb2c89cf557062a7fd356aa23457b3c6ef0e06b`）是对"中间态文件"计算的哈希：
  D0 脚本先写入含 `sha256_utf8: null` 的文件，随后读取该中间文件计算哈希，最后回填哈希再写回。
  因此该字段值与 D0 最终文件全文哈希（`a5d8d81611249cb076055a6a560b997996768a8f42fcea20041d607f8877f817`）不一致。
- 本门判定：**模板字符串本身**（`The answer is <answer>.`，SHA256=`c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc`）可唯一恢复，且全部 979 个 pair 的渲染候选均与模板一致，
  故 candidate template 的继承一致性与可恢复性成立；D0 的字段级哈希记录缺陷不影响 D0 冻结数据或本门。

## 模型评分记录

```text
dev_model_scored = true
train_model_scored = false
final_reserve_model_scored = false
```

final-reserve 文本未写入本轮任何文件（dev-only 提取即刻丢弃非 dev 行）。
