# future_h1_data_access_boundary.md

## 后续 PopQA H1 development 行为资格门的数据访问边界

- E0-R2 仅批准 PopQA 进入后续 Qwen H1 **development** 行为资格门。
- 下一轮（H1 dev）只能读取 **dev** 文本（E0-R1 dev split 的 2,815 组）。
- **train / final-reserve 文本不得读取、评分或缓存**；final-reserve 仅可在未来的冻结后一次性确认中接触（需另行授权协议）。
- H1 未通过时，不得进入 hidden state / Probe / monitor 阶段。
- H1 通过后，也必须**先单独请求下一阶段协议**，不得自动进入 H2 或任何 hidden-state 实验。
- 主指标按 group micro-average；relation 分层仅作描述性诊断，仅 `n >= 30` 的 relation 允许带 CI 的报告，`n < 30` 只报样本数。
