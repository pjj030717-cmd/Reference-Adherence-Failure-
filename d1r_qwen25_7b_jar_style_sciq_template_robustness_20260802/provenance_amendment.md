# provenance_amendment.md

## 背景：D0 `candidate_rendering_spec.json` 的 hash 缺陷

D0（`e01c6r2a…` 之外，即 `d0_jar_style_sciq_data_qualification_20260802`）生成 `candidate_rendering_spec.json` 时，
脚本流程为：

1. 先写入含 `"sha256_utf8": null` 占位的中间文件；
2. 读取该中间文件全文计算 SHA256；
3. 将算得的哈希回填进 JSON 后再写回最终文件。

因此最终文件中的 `sha256_utf8` 字段值（`d41ad577206de5b7c65465259bb2c89cf557062a7fd356aa23457b3c6ef0e06b`）
是对"中间态文件"的哈希，而**不是对最终文件全文**的哈希。最终文件全文的 SHA256 为 `a5d8d81611249cb076055a6a560b997996768a8f42fcea20041d607f8877f817`。

## 该缺陷是否影响样本内容？

**不影响。** 判断依据：

1. D0 模板字段为 `"template": "The answer is <answer>."`，模板**字符串**是唯一的内容依据；
2. `preliminary_swap_pairs.jsonl` 中全部 979 组 pair 的 `c_o` / `c_s` 均严格符合该模板（已验证 0 违例）；
3. D1 中全部 195 dev group × 4 cell 的评分均使用该模板渲染的候选，且 D1 四格行为（OO=1.000, OS=1.000, SO=0.928, SS=0.241）已验证。

即：模板字符串与其 UTF-8 SHA256（`c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc`）可唯一恢复，
行级四格结果均基于该字符串，故 D0 hash 记录缺陷不构成 D1-R 的继承失败。

## D2 后续应以何种 hash 继承？

- **继承模板字符串，而非文件的 `sha256_utf8` 字段**：
  - `template = "The answer is <answer>."`
  - `template_utf8_sha256 = c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc`
- 若需对 `candidate_rendering_spec.json` 文件本身固化，应以最终文件全文哈希 `a5d8d81611249cb076055a6a560b997996768a8f42fcea20041d607f8877f817` 为准；
- 不得以 `sha256_utf8` 字段值（`d41ad577…`）作为继承依据。

## 结论

- D1-R 唯一继承依据：模板原始字符串 + 模板字符串 UTF-8 SHA256 + D1 已验证的行级四格结果；
- D0 文件不修改；该缺陷仅作透明记录。
