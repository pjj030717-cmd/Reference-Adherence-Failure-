# lexical_comparator_spec.md

## 定义（在行为结果之前固定）

### B_slot_oracle —— 构造 oracle，不可当可部署 Judge
- 已知模板与渲染后的 Candidate，从唯一的 `<answer>` 占位符位置恢复答案 a
  （`a = candidate` 去掉模板 prefix/suffix）。
- 预测规则：`normalize(a) == normalize(r)` → Accept，否则 Reject。
- `normalize` = NFKC + trim + 空白折叠 + 小写。

### B_exact_match —— 简单 lexical comparator（非 oracle）
- 不读取占位符；将 Candidate 按 token 化（小写词形）。
- 预测规则：`normalize(r)` 作为完整 token/span（连续子序列）出现在 Candidate 中 → Accept，否则 Reject。

## 诚实定位

- 若规则 comparator 在此受控任务上表现强，说明该任务并不要求我们证明 Probe 优于字符串匹配；
  本轮研究的问题仍是冻结 LLM Judge 为何未遵从已给 Reference，以及该失效是否可由 Candidate 前状态预测。
- 不得因 lexical comparator 很强而删除、重写或否定 Judge 行为结果。
