# inheritance_audit.md

## 0.1 D0 继承审计

| 审计项 | 值 | 通过 |
|---|---|---|
| D0 最终标签 | jar_style_sciq_data_qualification_feasible | ✓ |
| D0 dev group 数 | 195 | ✓ |
| D0 split seed | 20260802 | ✓ |
| D0 dev split SHA256 | 8be6f6f3450376cb90c597ad0fe9edf2ea422101501900c13c5604db61e4fd35 | ✓ |
| dev 195 组与 D0 split 索引一致 | 195 pairs, set-equal | ✓ |
| swap 映射唯一性（r_o != r_s，swap 源 != 本组） | 195/195 OK | ✓ |
| T0 Candidate 渲染语义（candidate 唯一由冻结 r_o/r_s + T0 模板构成） | 780-cell OK | ✓ |
| D0 冻结 c_o/c_s 字段复核（dev-only 文件） | 390 OK | ✓ |
| 四格期望标签映射 | OO=A, OS=B, SO=B, SS=A | ✓ |
| T0/T1/T2 模板 SHA256 复核 | T0/T1/T2 全部一致 | ✓ |

- D0 最终标签：`jar_style_sciq_data_qualification_feasible`
- dev split：195 个 source group（D0 `fixed_split_indices.json`，seed 20260802）
- swap 映射来源：D0 coarse-form-controlled Random Swap（seed 20260802），经 D1 dev-only `_dev_pairs.jsonl` 流式继承。
- 四格构造：OO=(r_o,c_o), OS=(r_o,c_s), SO=(r_s,c_o), SS=(r_s,c_s)；期望标签 OO=A, OS=B, SO=B, SS=A。
- T0 Candidate 渲染：`candidate = "The answer is " + 冻结r + "."`（NFKC/trim/空白归一化，大小写不变）。
- 本轮目录不含 train / final-reserve 完整题目文本（仅经 D1 dev-only 文件流式读取 195 组 dev）。

## 0.2 模板继承（详见 prompt_semantic_inheritance_audit.md）

- 唯一序列化变化：Qwen chat template → Mistral `apply_chat_template`。
- system/user 内容、字段顺序、verdict、continuation、teacher-forced 位置逐字继承。
