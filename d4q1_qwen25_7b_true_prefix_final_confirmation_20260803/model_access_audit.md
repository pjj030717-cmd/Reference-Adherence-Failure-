# model_access_audit.md

## 模型

- 路径：`/root/autodl-tmp/models/Qwen2.5-7B-Instruct`
- 说明：`Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`（继承 D2-R1 manifest 记录）。
- 精度：BF16；`eval()`；`torch.inference_mode()`；`batch_size=1`；CUDA。

## 访问范围

| 阶段 | 前向数 | 数据 |
|---|---|---|
| Phase 1.1 A/B 回归 | 24 | 合成 pairs（无 D0 group） |
| Phase 1.2 合同 | 196 prefix + 196 完整 SS | 允许 final 组 |
| Phase 1.2 重复审计 | 60×2 prefix | 允许 final 组抽样 |
| Phase 2.1 标签生成 | 196 完整 SS | 允许 final 组（唯一正式评分） |

- 未对隔离 group（`0075758e…`）做任何前向、评分或 hidden-state 读取。
- 未加载 Mistral；未运行 activation intervention / prompt baseline / T1/T2 / 其他 cell。
- 未修改 D2-R1 或其他既有目录的任何文件。
